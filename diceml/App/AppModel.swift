import Foundation
import Observation

struct DatasetInfo: Identifiable, Equatable {
    let id: UUID
    let name: String
    let diceCount: Int
    let contextLength: Int
    let generation: Int
    let modelGeneration: Int
    let runID: UUID
    let historyRevision: Int
    let settingsRevision: Int
    let nextOrdinal: Int
    let currentSessionID: UUID

    init(_ record: DatasetRecord) {
        id = record.id
        name = record.name
        diceCount = record.diceCount
        contextLength = record.contextLength
        generation = record.generation
        modelGeneration = record.modelGeneration
        runID = record.activeRunID
        historyRevision = record.historyRevision
        settingsRevision = record.settingsRevision
        nextOrdinal = record.nextOrdinal
        currentSessionID = record.currentSessionID
    }

    var schema: FeatureSchema { FeatureSchema(diceCount: diceCount, contextLength: contextLength) }

    var identity: ModelIdentity {
        ModelIdentity(datasetID: id, datasetGeneration: generation, modelGeneration: modelGeneration, runID: runID)
    }
}

struct HistoryItem: Identifiable, Equatable {
    let id: UUID
    let ordinal: Int
    let values: [Int]
    let observedAt: Date
    let isEdited: Bool

    var sum: Int { values.reduce(0, +) }
}

/// Consecutive rolls from one session, newest first.
struct HistorySection: Identifiable, Equatable {
    let id: UUID
    var items: [HistoryItem]
}

/// Coordinates history, the pending prediction target, entry drafts, and training.
/// All context transitions run on the main actor, so they are serialized.
@MainActor
@Observable
final class AppModel {
    enum LoadState: Equatable {
        case loading
        case failed(String)
        case ready
    }

    enum CardState: Equatable {
        case empty
        case updating
        case unavailable
        case published(ForecastSnapshot)

        var forecast: ForecastSnapshot? {
            if case .published(let forecast) = self { forecast } else { nil }
        }
    }

    private(set) var loadState: LoadState = .loading
    private(set) var dataset: DatasetInfo?
    private(set) var datasets: [DatasetInfo] = []
    private(set) var historySections: [HistorySection] = []
    private(set) var observationCount = 0
    private(set) var card: CardState = .empty
    private(set) var round: RoundContext?
    private(set) var isTraining = false
    private(set) var trainedPairs = 0
    private(set) var evaluation: [TimingCategory: EvaluationMetrics] = [:]
    /// Walk-forward replay of the current predictor over existing history.
    private(set) var replay: EvaluationMetrics?
    var errorMessage: String?

    private let storeFactory: @MainActor () throws -> DiceStore
    private let publicationDeadline: Duration
    private let trainer = SequenceTrainer()
    private var store: DiceStore?
    private var rolls: [RollValue] = []
    private var publicationOpen = false
    private var draft: EntryDraft?
    private var retiredTargets: Set<UUID> = []
    private var retiredSubmissions: Set<UUID> = []
    private var deadlineTask: Task<Void, Never>?
    private var trainingTask: Task<Void, Never>?
    private var trainingToken = UUID()
    /// The target whose terminal forecast source has been chosen. Exactly one source is chosen per target.
    private var decidedTarget: UUID?
    private var forecastTask: Task<Void, Never>?

    init(
        storeFactory: @escaping @MainActor () throws -> DiceStore = { try DiceStore.openDefault() },
        publicationDeadline: Duration = .seconds(2)
    ) {
        self.storeFactory = storeFactory
        self.publicationDeadline = publicationDeadline
    }

    // MARK: Loading

    func load() {
        cancelOutstandingWork()
        store = nil
        loadState = .loading
        do {
            store = try storeFactory()
            try reload()
            loadState = .ready
            beginRound()
        } catch {
            fail(error)
        }
    }

    /// Unknown history is a recovery state: never present it as empty or fall back to random output.
    private func fail(_ error: Error) {
        cancelOutstandingWork()
        round = nil
        card = .empty
        historySections = []
        loadState = .failed(error.localizedDescription)
    }

    private func reload() throws {
        guard let store else { return }
        let active = try store.activeDataset()
        let records = try store.rolls(of: active)
        let predictions = try store.predictions(of: active)

        dataset = DatasetInfo(active)
        datasets = try store.datasets().map(DatasetInfo.init)
        rolls = records.map(\.value)
        observationCount = records.count
        trainedPairs = (try? store.checkpoint(datasetID: active.id))?.trainedPairCount ?? 0

        var sections: [HistorySection] = []
        for record in records.reversed() {
            let item = HistoryItem(
                id: record.id,
                ordinal: record.ordinal,
                values: record.values,
                observedAt: record.observedAt,
                isEdited: record.revision > 0
            )
            if sections.last?.id == record.sessionID {
                sections[sections.count - 1].items.append(item)
            } else {
                sections.append(HistorySection(id: record.sessionID, items: [item]))
            }
        }
        historySections = sections

        let actuals = Dictionary(uniqueKeysWithValues: records.map { ($0.id, $0.values) })
        evaluation = Evaluator.evaluate(predictions.compactMap { prediction in
            guard !prediction.revised,
                  prediction.mode != PredictionMode.random.rawValue,
                  let actualID = prediction.matchedActualID,
                  let actual = actuals[actualID],
                  let timing = prediction.timing.flatMap(TimingCategory.init(rawValue:))
            else { return nil }
            return ScoredPrediction(
                values: prediction.values,
                probabilities: prediction.probabilities,
                baseline: prediction.baseline,
                actual: actual,
                timing: timing
            )
        })
    }

    // MARK: Prediction rounds

    /// Opens a new pending target for the current history, or restores a persisted forecast for it.
    private func beginRound() {
        deadlineTask?.cancel()
        forecastTask?.cancel()
        decidedTarget = nil
        guard loadState == .ready, let store, let dataset else {
            round = nil
            card = .empty
            return
        }

        let context = RoundContext(
            datasetID: dataset.id,
            datasetGeneration: dataset.generation,
            modelGeneration: dataset.modelGeneration,
            runID: dataset.runID,
            sessionID: dataset.currentSessionID,
            targetID: UUID(),
            expectedOrdinal: dataset.nextOrdinal,
            historyRevision: dataset.historyRevision,
            settingsRevision: dataset.settingsRevision
        )
        publicationOpen = true

        if let restored = try? store.pendingPrediction(matching: context) {
            round = restored.context
            card = .published(restored)
            refreshReplay()
            startTraining(for: restored.context)
            return
        }

        round = context
        // Only a successful load that found zero observations permits random output.
        guard !rolls.isEmpty else {
            var generator = SystemRandomNumberGenerator()
            decidedTarget = context.targetID
            replay = nil
            publish(Forecaster.random(diceCount: dataset.diceCount, using: &generator), for: context)
            return
        }

        card = .updating
        let schema = dataset.schema
        let cached = try? store.checkpoint(datasetID: dataset.id)
        if let cached, cached.identity == dataset.identity, cached.schema == schema, cached.covers(rolls) {
            publishLearned(for: context, checkpoint: cached)
            return
        }

        let hasLagContext = SequenceFeatures.currentLags(
            rolls, sessionID: context.sessionID, expectedOrdinal: context.expectedOrdinal, schema: schema) != nil
        startTraining(for: context)
        guard hasLagContext, SequenceFeatures.validPairCount(rolls) > 0 else {
            publishLearned(for: context, checkpoint: nil)
            return
        }

        // Publish frequencies at the deadline unless the sequence fit finishes first.
        let deadline = publicationDeadline
        deadlineTask = Task { [weak self] in
            try? await Task.sleep(for: deadline)
            guard !Task.isCancelled else { return }
            self?.publishLearned(for: context, checkpoint: nil)
        }
    }

    private func startTraining(for context: RoundContext) {
        trainingTask?.cancel()
        trainingToken = UUID()
        guard let store, let dataset, SequenceFeatures.validPairCount(rolls) > 0 else {
            isTraining = false
            return
        }
        // A model-cache read error falls back to rebuilding from actual history.
        let base = try? store.checkpoint(datasetID: dataset.id)
        let job = TrainingJob(identity: dataset.identity, schema: dataset.schema, rolls: rolls, base: base)
        if let base, base.identity == job.identity, base.schema == job.schema, base.covers(rolls) {
            isTraining = false
            return
        }

        let token = UUID()
        trainingToken = token
        isTraining = true
        trainingTask = Task { [weak self, trainer] in
            do {
                let checkpoint = try await trainer.train(job)
                self?.finishTraining(checkpoint, for: context, token: token)
            } catch {
                self?.trainingFailed(for: context, token: token)
            }
        }
    }

    private func finishTraining(_ checkpoint: Checkpoint, for context: RoundContext, token: UUID) {
        guard token == trainingToken else { return }
        isTraining = false
        install(checkpoint)
        publishLearned(for: context, checkpoint: checkpoint)
    }

    private func trainingFailed(for context: RoundContext, token: UUID) {
        guard token == trainingToken else { return }
        isTraining = false
        publishLearned(for: context, checkpoint: nil)
    }

    /// Caches a checkpoint only if it belongs to the active run and its trained history is unchanged,
    /// and never lets an older checkpoint replace newer progress from the same lineage.
    private func install(_ checkpoint: Checkpoint) {
        guard let store, let dataset,
              checkpoint.identity == dataset.identity,
              checkpoint.schema == dataset.schema,
              checkpoint.isValidPrefix(of: rolls)
        else { return }
        if let existing = try? store.checkpoint(datasetID: dataset.id),
           existing.identity == checkpoint.identity,
           existing.schema == checkpoint.schema,
           existing.isValidPrefix(of: rolls),
           existing.trainedThroughOrdinal > checkpoint.trainedThroughOrdinal {
            return
        }
        do {
            try store.saveCheckpoint(checkpoint)
            trainedPairs = checkpoint.trainedPairCount
        } catch {
            // The checkpoint is a rebuildable cache; history and the forecast path are unaffected.
        }
    }

    /// Chooses the terminal source for a target once: the sequence-aware forecast when a checkpoint
    /// covering the current history is ready, otherwise learned frequencies. The forecast is computed
    /// off the main actor and published only if the target is still open when it finishes.
    private func publishLearned(for context: RoundContext, checkpoint: Checkpoint?) {
        guard canPublish(for: context), decidedTarget != context.targetID, let dataset, !rolls.isEmpty else { return }
        decidedTarget = context.targetID
        deadlineTask?.cancel()
        let usable = checkpoint.flatMap { $0.identity == dataset.identity && $0.covers(rolls) ? $0 : nil }
        let rolls = rolls
        let schema = dataset.schema
        forecastTask = Task { [weak self] in
            let result = await Task.detached(priority: .userInitiated) {
                Forecaster.learned(
                    rolls: rolls,
                    sessionID: context.sessionID,
                    expectedOrdinal: context.expectedOrdinal,
                    schema: schema,
                    checkpoint: usable
                )
            }.value
            guard !Task.isCancelled, let self, self.round == context else { return }
            self.replay = result.replay
            self.publish(result, for: context)
        }
    }

    /// Recomputes the walk-forward replay for a restored forecast without publishing anything.
    private func refreshReplay() {
        guard let store, let dataset, let round, !rolls.isEmpty else {
            replay = nil
            return
        }
        let cached = try? store.checkpoint(datasetID: dataset.id)
        let usable = cached.flatMap { $0.identity == dataset.identity && $0.covers(rolls) ? $0 : nil }
        let rolls = rolls
        let schema = dataset.schema
        forecastTask = Task { [weak self] in
            let result = await Task.detached(priority: .utility) {
                Forecaster.learned(
                    rolls: rolls,
                    sessionID: round.sessionID,
                    expectedOrdinal: round.expectedOrdinal,
                    schema: schema,
                    checkpoint: usable
                )
            }.value
            guard !Task.isCancelled, let self, self.round == round else { return }
            self.replay = result.replay
        }
    }

    /// Single-publication rule: one terminal, persisted forecast per open target.
    private func publish(_ result: ForecastResult, for context: RoundContext) {
        guard canPublish(for: context), let store else { return }
        let snapshot = ForecastSnapshot(
            id: UUID(),
            context: context,
            values: result.values,
            probabilities: result.probabilities,
            baseline: result.baseline,
            mode: result.mode,
            trainedPairs: result.trainedPairs,
            blendWeight: result.blendWeight,
            evidence: result.evidence,
            publishedAt: .now
        )
        do {
            try store.savePrediction(snapshot)
            deadlineTask?.cancel()
            card = .published(snapshot)
        } catch {
            card = .unavailable
            errorMessage = error.localizedDescription
        }
    }

    private func canPublish(for context: RoundContext) -> Bool {
        round == context
            && publicationOpen
            && !retiredTargets.contains(context.targetID)
            && card.forecast?.context.targetID != context.targetID
    }

    // MARK: Entry

    /// Seals the current target and pins its published forecast, if any.
    func beginEntry() -> EntryDraft? {
        guard loadState == .ready, let round else { return nil }
        if let draft { return draft }
        publicationOpen = false
        deadlineTask?.cancel()
        let forecast = card.forecast.flatMap { $0.context == round ? $0 : nil }
        if forecast == nil { card = .unavailable }
        let draft = EntryDraft(submissionID: UUID(), context: round, predictionID: forecast?.id)
        self.draft = draft
        return draft
    }

    /// Cancels an open draft. A published forecast stays unchanged; an unpublished target is retired
    /// and replaced by a fresh one for the same history.
    func cancelEntry() {
        guard let draft else { return }
        self.draft = nil
        retiredSubmissions.insert(draft.submissionID)
        if draft.predictionID == nil {
            retiredTargets.insert(draft.context.targetID)
            beginRound()
        }
    }

    func saveEntry(values: [Int], attested: Bool) throws {
        guard let draft, !retiredSubmissions.contains(draft.submissionID), let store else {
            throw StoreError.staleContext
        }
        try store.saveActual(
            draft: draft,
            values: values,
            timing: draft.predictionID == nil ? nil : (attested ? .attested : .preEntry),
            source: .manual
        )
        self.draft = nil
        retiredSubmissions.insert(draft.submissionID)
        retiredTargets.insert(draft.context.targetID)
        refreshAndBeginRound()
    }

    // MARK: History and settings changes

    func editRoll(id: UUID, values: [Int]) throws {
        guard let store, draft == nil else { throw StoreError.staleContext }
        cancelOutstandingWork()
        do {
            try store.editRoll(id: id, values: values)
        } catch {
            refreshAndBeginRound()
            throw error
        }
        retireCurrentTarget()
        refreshAndBeginRound()
    }

    func deleteRoll(id: UUID) {
        perform { try $0.deleteRoll(id: id) }
    }

    func startSession() {
        guard let dataset else { return }
        perform { try $0.startSession(datasetID: dataset.id) }
    }

    func setContextLength(_ length: Int) {
        guard let dataset else { return }
        perform { try $0.setContextLength(length, datasetID: dataset.id) }
    }

    func selectDataset(_ id: UUID) {
        guard id != dataset?.id else { return }
        // Switching datasets leaves the previous dataset's pending forecast valid for its own target.
        perform(retiringTarget: false) { try $0.select(datasetID: id) }
    }

    /// Default name for the next table, such as "Table 2".
    var suggestedTableName: String {
        String(localized: "Table \(datasets.count + 1)")
    }

    func createDataset(name: String, diceCount: Int) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let finalName = trimmed.isEmpty ? suggestedTableName : trimmed
        perform { try $0.createDataset(name: finalName, diceCount: diceCount) }
    }

    func trainNewModel() {
        guard let dataset else { return }
        perform { try $0.trainNewModel(datasetID: dataset.id) }
    }

    func deleteCurrentTable() {
        guard let dataset else { return }
        perform { try $0.deleteDataset(id: dataset.id) }
    }

    func clearObservations() {
        guard let dataset else { return }
        perform { try $0.clearObservations(datasetID: dataset.id) }
    }

    /// Invalidates in-flight work before a context change, applies it durably, then opens a new target.
    /// On failure the previous durable state remains and its forecast path resumes.
    private func perform(retiringTarget: Bool = true, _ change: (DiceStore) throws -> Void) {
        guard let store, draft == nil else { return }
        cancelOutstandingWork()
        do {
            try change(store)
            if retiringTarget { retireCurrentTarget() }
        } catch {
            errorMessage = error.localizedDescription
        }
        refreshAndBeginRound()
    }

    private func retireCurrentTarget() {
        if let round { retiredTargets.insert(round.targetID) }
    }

    private func refreshAndBeginRound() {
        do {
            try reload()
            beginRound()
        } catch {
            fail(error)
        }
    }

    private func cancelOutstandingWork() {
        deadlineTask?.cancel()
        forecastTask?.cancel()
        trainingTask?.cancel()
        trainingToken = UUID()
        isTraining = false
    }
}
