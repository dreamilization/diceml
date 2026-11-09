import Foundation
import SwiftData

enum StoreError: LocalizedError, Equatable {
    case staleContext
    case invalidRoll
    case missingDataset
    case missingRoll

    var errorDescription: String? {
        switch self {
        case .staleContext: String(localized: "This roll belongs to an earlier round. Review it and add it again.")
        case .invalidRoll: String(localized: "Each die needs a value from 1 to 6.")
        case .missingDataset: String(localized: "The selected table could not be found.")
        case .missingRoll: String(localized: "That roll no longer exists.")
        }
    }
}

/// Owns the local SwiftData store. Every mutation is one serialized transaction that either
/// saves completely or rolls back, leaving the previous durable state intact.
@MainActor
final class DiceStore {
    let container: ModelContainer
    private var context: ModelContext { container.mainContext }

    init(container: ModelContainer) {
        self.container = container
        container.mainContext.autosaveEnabled = false
    }

    /// Opens the on-device store without CloudKit sync. Failures propagate: an unreadable store is
    /// never replaced by an empty one.
    static func openDefault() throws -> DiceStore {
        let directory = URL.applicationSupportDirectory
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return try open(at: directory.appending(path: "DiceML.store"))
    }

    /// Opens (and if needed migrates) the store at `url`.
    static func open(at url: URL) throws -> DiceStore {
        let schema = Schema(versionedSchema: DiceSchemaV2.self)
        let configuration = ModelConfiguration(schema: schema, url: url, cloudKitDatabase: .none)
        let container = try ModelContainer(
            for: schema,
            migrationPlan: DiceMigrationPlan.self,
            configurations: configuration
        )
        return DiceStore(container: container)
    }

    static func inMemory() throws -> DiceStore {
        let schema = Schema(versionedSchema: DiceSchemaV2.self)
        let configuration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: true, cloudKitDatabase: .none)
        return DiceStore(container: try ModelContainer(for: schema, configurations: configuration))
    }

    private func transaction<T>(_ body: () throws -> T) throws -> T {
        do {
            let result = try body()
            try context.save()
            return result
        } catch {
            context.rollback()
            throw error
        }
    }

    // MARK: Reading

    func datasets() throws -> [DatasetRecord] {
        try context.fetch(FetchDescriptor<DatasetRecord>(sortBy: [SortDescriptor(\.createdAt)]))
    }

    /// The most recently selected dataset. A successfully loaded store with no datasets gets the default one.
    func activeDataset() throws -> DatasetRecord {
        var descriptor = FetchDescriptor<DatasetRecord>(sortBy: [SortDescriptor(\.lastSelectedAt, order: .reverse)])
        descriptor.fetchLimit = 1
        if let dataset = try context.fetch(descriptor).first { return dataset }
        return try transaction { insertDataset(name: String(localized: "Table \(1)"), diceCount: DiceRules.defaultDiceCount) }
    }

    func dataset(id: UUID) throws -> DatasetRecord {
        let descriptor = FetchDescriptor<DatasetRecord>(predicate: #Predicate { $0.id == id })
        guard let dataset = try context.fetch(descriptor).first else { throw StoreError.missingDataset }
        return dataset
    }

    func rolls(of dataset: DatasetRecord) throws -> [RollRecord] {
        let id = dataset.id
        let generation = dataset.generation
        return try context.fetch(FetchDescriptor<RollRecord>(
            predicate: #Predicate { $0.datasetID == id && $0.datasetGeneration == generation },
            sortBy: [SortDescriptor(\.ordinal)]
        ))
    }

    func predictions(of dataset: DatasetRecord) throws -> [PredictionRecord] {
        let id = dataset.id
        let generation = dataset.generation
        return try context.fetch(FetchDescriptor<PredictionRecord>(
            predicate: #Predicate { $0.datasetID == id && $0.datasetGeneration == generation },
            sortBy: [SortDescriptor(\.publishedAt)]
        ))
    }

    /// A persisted pending forecast whose full context still matches, so it survives relaunch unchanged.
    func pendingPrediction(matching context: RoundContext) throws -> ForecastSnapshot? {
        try pendingPredictions(datasetID: context.datasetID)
            .first { $0.context.sharesState(with: context) }?
            .snapshot
    }

    func checkpoint(datasetID: UUID) throws -> Checkpoint? {
        try modelState(datasetID: datasetID)?.checkpoint
    }

    // MARK: Writing

    @discardableResult
    func createDataset(name: String, diceCount: Int) throws -> DatasetRecord {
        guard DiceRules.diceCountRange.contains(diceCount) else { throw StoreError.invalidRoll }
        return try transaction { insertDataset(name: name, diceCount: diceCount) }
    }

    func select(datasetID: UUID) throws {
        try transaction { try dataset(id: datasetID).lastSelectedAt = .now }
    }

    /// Records one actual roll, links its pinned forecast, and advances the history revision atomically.
    /// Retrying the same submission returns the original roll instead of creating a duplicate.
    @discardableResult
    func saveActual(
        draft: EntryDraft,
        values: [Int],
        timing: TimingCategory?,
        source: InputSource,
        observedAt: Date = .now
    ) throws -> UUID {
        try transaction {
            let submissionID = draft.submissionID
            let existing = FetchDescriptor<RollRecord>(predicate: #Predicate { $0.submissionID == submissionID })
            if let roll = try context.fetch(existing).first { return roll.id }

            let dataset = try self.dataset(id: draft.context.datasetID)
            guard matches(dataset, draft.context) else { throw StoreError.staleContext }
            guard DiceRules.isValid(values, diceCount: dataset.diceCount) else { throw StoreError.invalidRoll }

            let roll = RollRecord(
                submissionID: submissionID,
                context: draft.context,
                ordinal: dataset.nextOrdinal,
                values: values,
                source: source,
                observedAt: observedAt
            )
            context.insert(roll)

            for prediction in try pendingPredictions(datasetID: dataset.id) {
                if prediction.id == draft.predictionID, prediction.targetID == draft.context.targetID {
                    prediction.matchedActualID = roll.id
                    prediction.timing = (timing ?? .preEntry).rawValue
                } else {
                    prediction.retired = true
                }
            }
            dataset.nextOrdinal += 1
            dataset.historyRevision += 1
            return roll.id
        }
    }

    /// Persists a forecast only while its context is still the dataset's current state.
    func savePrediction(_ snapshot: ForecastSnapshot) throws {
        try transaction {
            let dataset = try self.dataset(id: snapshot.context.datasetID)
            guard matches(dataset, snapshot.context) else { throw StoreError.staleContext }
            for pending in try pendingPredictions(datasetID: dataset.id) { pending.retired = true }
            context.insert(PredictionRecord(snapshot))
        }
    }

    func editRoll(id: UUID, values: [Int]) throws {
        try transaction {
            let roll = try self.roll(id: id)
            let dataset = try self.dataset(id: roll.datasetID)
            guard DiceRules.isValid(values, diceCount: dataset.diceCount) else { throw StoreError.invalidRoll }
            guard roll.values != values else { return }
            roll.values = values
            roll.revision += 1
            roll.editedAt = .now
            try invalidateHistory(of: dataset, fromOrdinal: roll.ordinal)
        }
    }

    /// Deletes a roll's values. Its ordinal is never reused, which leaves a sequence gap.
    func deleteRoll(id: UUID) throws {
        try transaction {
            let roll = try self.roll(id: id)
            let dataset = try self.dataset(id: roll.datasetID)
            let ordinal = roll.ordinal
            context.delete(roll)
            try invalidateHistory(of: dataset, fromOrdinal: ordinal)
        }
    }

    func startSession(datasetID: UUID) throws {
        try transaction {
            let dataset = try self.dataset(id: datasetID)
            let session = SessionRecord(datasetID: datasetID)
            context.insert(session)
            dataset.currentSessionID = session.id
            try retirePending(datasetID: datasetID)
        }
    }

    func setContextLength(_ length: Int, datasetID: UUID) throws {
        guard DiceRules.contextLengthRange.contains(length) else { return }
        try transaction {
            let dataset = try self.dataset(id: datasetID)
            guard dataset.contextLength != length else { return }
            dataset.contextLength = length
            dataset.settingsRevision += 1
            try deleteModelState(datasetID: datasetID)
            try retirePending(datasetID: datasetID)
        }
    }

    /// Starts a new identified training run on retained observations.
    func trainNewModel(datasetID: UUID) throws {
        try transaction {
            let dataset = try self.dataset(id: datasetID)
            dataset.modelGeneration += 1
            dataset.activeRunID = UUID()
            try deleteModelState(datasetID: datasetID)
            try retirePending(datasetID: datasetID)
        }
    }

    /// Removes every observation, prediction, session, and model state of one dataset, then
    /// advances its generations and opens a fresh session. Other datasets are untouched.
    func clearObservations(datasetID: UUID) throws {
        try transaction {
            let dataset = try self.dataset(id: datasetID)
            try context.delete(model: RollRecord.self, where: #Predicate { $0.datasetID == datasetID })
            try context.delete(model: PredictionRecord.self, where: #Predicate { $0.datasetID == datasetID })
            try context.delete(model: SessionRecord.self, where: #Predicate { $0.datasetID == datasetID })
            try deleteModelState(datasetID: datasetID)

            let session = SessionRecord(datasetID: datasetID)
            context.insert(session)
            dataset.currentSessionID = session.id
            dataset.generation += 1
            dataset.modelGeneration += 1
            dataset.activeRunID = UUID()
            dataset.historyRevision += 1
        }
    }

    /// Deletes a table and everything recorded for it. Another table becomes active; if none remain,
    /// a fresh default table is created so the app always has somewhere to record.
    func deleteDataset(id datasetID: UUID) throws {
        try transaction {
            let dataset = try self.dataset(id: datasetID)
            try context.delete(model: RollRecord.self, where: #Predicate { $0.datasetID == datasetID })
            try context.delete(model: PredictionRecord.self, where: #Predicate { $0.datasetID == datasetID })
            try context.delete(model: SessionRecord.self, where: #Predicate { $0.datasetID == datasetID })
            try deleteModelState(datasetID: datasetID)
            context.delete(dataset)

            let remaining = try context.fetch(FetchDescriptor<DatasetRecord>(
                predicate: #Predicate { $0.id != datasetID },
                sortBy: [SortDescriptor(\.lastSelectedAt, order: .reverse)]
            ))
            if let next = remaining.first {
                next.lastSelectedAt = .now
            } else {
                _ = insertDataset(name: String(localized: "Table \(1)"), diceCount: DiceRules.defaultDiceCount)
            }
        }
    }

    /// Installs a checkpoint only for the dataset's active generation and run.
    func saveCheckpoint(_ checkpoint: Checkpoint) throws {
        try transaction {
            let dataset = try self.dataset(id: checkpoint.identity.datasetID)
            guard dataset.generation == checkpoint.identity.datasetGeneration,
                  dataset.modelGeneration == checkpoint.identity.modelGeneration,
                  dataset.activeRunID == checkpoint.identity.runID
            else { throw StoreError.staleContext }
            if let state = try modelState(datasetID: dataset.id) {
                state.update(from: checkpoint)
            } else {
                context.insert(ModelStateRecord(checkpoint))
            }
        }
    }

    // MARK: Helpers

    private func insertDataset(name: String, diceCount: Int) -> DatasetRecord {
        let session = SessionRecord(datasetID: UUID())
        let dataset = DatasetRecord(name: name, diceCount: diceCount, sessionID: session.id)
        session.datasetID = dataset.id
        context.insert(dataset)
        context.insert(session)
        return dataset
    }

    private func matches(_ dataset: DatasetRecord, _ context: RoundContext) -> Bool {
        dataset.generation == context.datasetGeneration
            && dataset.modelGeneration == context.modelGeneration
            && dataset.activeRunID == context.runID
            && dataset.currentSessionID == context.sessionID
            && dataset.nextOrdinal == context.expectedOrdinal
            && dataset.historyRevision == context.historyRevision
            && dataset.settingsRevision == context.settingsRevision
    }

    private func roll(id: UUID) throws -> RollRecord {
        let descriptor = FetchDescriptor<RollRecord>(predicate: #Predicate { $0.id == id })
        guard let roll = try context.fetch(descriptor).first else { throw StoreError.missingRoll }
        return roll
    }

    private func pendingPredictions(datasetID: UUID) throws -> [PredictionRecord] {
        try context.fetch(FetchDescriptor<PredictionRecord>(predicate: #Predicate {
            $0.datasetID == datasetID && $0.matchedActualID == nil && !$0.retired
        }))
    }

    private func retirePending(datasetID: UUID) throws {
        for prediction in try pendingPredictions(datasetID: datasetID) { prediction.retired = true }
    }

    private func modelState(datasetID: UUID) throws -> ModelStateRecord? {
        try context.fetch(FetchDescriptor<ModelStateRecord>(predicate: #Predicate { $0.datasetID == datasetID })).first
    }

    private func deleteModelState(datasetID: UUID) throws {
        try context.delete(model: ModelStateRecord.self, where: #Predicate { $0.datasetID == datasetID })
    }

    /// After an edit or deletion: bump the revision, drop the model cache, retire the pending target, and
    /// mark as revised every forecast whose target was the changed roll or whose lag window contained it.
    /// Earlier forecasts, and later ones that never saw the roll as direct input, remain valid pre-target
    /// forecasts and keep their scores.
    private func invalidateHistory(of dataset: DatasetRecord, fromOrdinal ordinal: Int) throws {
        let id = dataset.id
        // Use the largest allowed context so forecasts made under an earlier, longer setting are covered.
        let lastAffected = ordinal + DiceRules.contextLengthRange.upperBound
        let affected = try context.fetch(FetchDescriptor<PredictionRecord>(predicate: #Predicate {
            $0.datasetID == id && $0.expectedOrdinal >= ordinal && $0.expectedOrdinal <= lastAffected
        }))
        for prediction in affected { prediction.revised = true }
        dataset.historyRevision += 1
        try deleteModelState(datasetID: id)
        try retirePending(datasetID: id)
    }
}
