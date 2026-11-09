import Foundation
import SwiftData
import Testing
@testable import diceml

@MainActor
struct StoreTests {
    private func round(_ store: DiceStore, _ dataset: DatasetRecord) -> RoundContext {
        RoundContext(
            datasetID: dataset.id,
            datasetGeneration: dataset.generation,
            modelGeneration: dataset.modelGeneration,
            runID: dataset.activeRunID,
            sessionID: dataset.currentSessionID,
            targetID: UUID(),
            expectedOrdinal: dataset.nextOrdinal,
            historyRevision: dataset.historyRevision,
            settingsRevision: dataset.settingsRevision
        )
    }

    @Test func defaultDatasetHasThreeDice() throws {
        let store = try DiceStore.inMemory()
        #expect(try store.activeDataset().diceCount == 3)
    }

    @Test func retriedSubmissionSavesOnce() throws {
        let store = try DiceStore.inMemory()
        let dataset = try store.activeDataset()
        let draft = EntryDraft(submissionID: UUID(), context: round(store, dataset), predictionID: nil)
        let first = try store.saveActual(draft: draft, values: [1, 2, 3], timing: nil, source: .manual)
        let retry = try store.saveActual(draft: draft, values: [1, 2, 3], timing: nil, source: .manual)
        #expect(first == retry)
        #expect(try store.rolls(of: dataset).count == 1)

        // Identical values in a genuine next roll are a separate observation.
        let next = EntryDraft(submissionID: UUID(), context: round(store, dataset), predictionID: nil)
        try store.saveActual(draft: next, values: [1, 2, 3], timing: nil, source: .manual)
        #expect(try store.rolls(of: dataset).map(\.ordinal) == [1, 2])
    }

    @Test func staleOrInvalidEntriesAreRejected() throws {
        let store = try DiceStore.inMemory()
        let dataset = try store.activeDataset()
        let stale = EntryDraft(submissionID: UUID(), context: round(store, dataset), predictionID: nil)
        try store.startSession(datasetID: dataset.id)
        #expect(throws: StoreError.staleContext) {
            try store.saveActual(draft: stale, values: [1, 2, 3], timing: nil, source: .manual)
        }
        let current = EntryDraft(submissionID: UUID(), context: round(store, dataset), predictionID: nil)
        #expect(throws: StoreError.invalidRoll) {
            try store.saveActual(draft: current, values: [1, 7, 3], timing: nil, source: .manual)
        }
        #expect(try store.rolls(of: dataset).isEmpty)
    }

    @Test func deletedOrdinalsAreNeverReused() throws {
        let store = try DiceStore.inMemory()
        let dataset = try store.activeDataset()
        for _ in 0..<3 {
            try store.saveActual(draft: EntryDraft(submissionID: UUID(), context: round(store, dataset), predictionID: nil),
                                 values: [4, 4, 4], timing: nil, source: .manual)
        }
        let middle = try store.rolls(of: dataset)[1]
        try store.deleteRoll(id: middle.id)
        try store.saveActual(draft: EntryDraft(submissionID: UUID(), context: round(store, dataset), predictionID: nil),
                             values: [4, 4, 4], timing: nil, source: .manual)
        let rolls = try store.rolls(of: dataset).map(\.value)
        #expect(rolls.map(\.ordinal) == [1, 3, 4])
        #expect(SequenceFeatures.validPairCount(rolls) == 1)
    }

    @Test func clearObservationsOnlyAffectsTheSelectedDataset() throws {
        let store = try DiceStore.inMemory()
        let dataset = try store.activeDataset()
        let other = try store.createDataset(name: "Other", diceCount: 2)
        try store.saveActual(draft: EntryDraft(submissionID: UUID(), context: round(store, dataset), predictionID: nil),
                             values: [1, 1, 1], timing: nil, source: .manual)
        try store.saveActual(draft: EntryDraft(submissionID: UUID(), context: round(store, other), predictionID: nil),
                             values: [2, 2], timing: nil, source: .manual)
        let generation = dataset.generation
        let run = dataset.activeRunID

        try store.clearObservations(datasetID: dataset.id)
        #expect(try store.rolls(of: dataset).isEmpty)
        #expect(dataset.generation > generation)
        #expect(dataset.activeRunID != run)
        #expect(try store.rolls(of: other).count == 1)
    }

    @Test func deletingATableRemovesOnlyItsHistory() throws {
        let store = try DiceStore.inMemory()
        let first = try store.activeDataset()
        let firstID = first.id
        try store.saveActual(draft: EntryDraft(submissionID: UUID(), context: round(store, first), predictionID: nil),
                             values: [1, 2, 3], timing: nil, source: .manual)
        let second = try store.createDataset(name: "Second", diceCount: 2)
        try store.saveActual(draft: EntryDraft(submissionID: UUID(), context: round(store, second), predictionID: nil),
                             values: [4, 4], timing: nil, source: .manual)

        try store.deleteDataset(id: second.id)
        let active = try store.activeDataset()
        #expect(active.id == firstID)
        #expect(try store.datasets().count == 1)
        #expect(try store.rolls(of: active).count == 1)

        // Deleting the last table leaves a fresh, empty default table.
        try store.deleteDataset(id: firstID)
        let replacement = try store.activeDataset()
        #expect(replacement.id != firstID)
        #expect(replacement.diceCount == 3)
        #expect(try store.rolls(of: replacement).isEmpty)
    }

    @Test func trainNewModelKeepsObservationsAndDropsTheCheckpoint() throws {
        let store = try DiceStore.inMemory()
        let dataset = try store.activeDataset()
        for faces in [[1, 2, 3], [2, 3, 4], [3, 4, 5]] {
            try store.saveActual(draft: EntryDraft(submissionID: UUID(), context: round(store, dataset), predictionID: nil),
                                 values: faces, timing: nil, source: .manual)
        }
        let identity = ModelIdentity(datasetID: dataset.id, datasetGeneration: dataset.generation,
                                     modelGeneration: dataset.modelGeneration, runID: dataset.activeRunID)
        let schema = FeatureSchema(diceCount: 3, contextLength: dataset.contextLength)
        let checkpoint = try SequenceTrainer.run(TrainingJob(
            identity: identity, schema: schema, rolls: try store.rolls(of: dataset).map(\.value), base: nil))
        try store.saveCheckpoint(checkpoint)
        #expect(try store.checkpoint(datasetID: dataset.id) != nil)

        try store.trainNewModel(datasetID: dataset.id)
        #expect(try store.rolls(of: dataset).count == 3)
        #expect(try store.checkpoint(datasetID: dataset.id) == nil)
        // A late checkpoint from the superseded run is rejected.
        #expect(throws: StoreError.staleContext) { try store.saveCheckpoint(checkpoint) }
    }
}

@MainActor
struct AppModelTests {
    private func readyModel(deadline: Duration = .seconds(2)) throws -> (AppModel, DiceStore) {
        let store = try DiceStore.inMemory()
        let model = AppModel(storeFactory: { store }, publicationDeadline: deadline)
        model.load()
        return (model, store)
    }

    /// Learned forecasts are computed off the main actor, so wait briefly for publication.
    private func publishedForecast(_ model: AppModel) async -> ForecastSnapshot? {
        for _ in 0..<300 where model.card.forecast == nil {
            try? await Task.sleep(for: .milliseconds(10))
        }
        return model.card.forecast
    }

    private func addRoll(_ model: AppModel, _ values: [Int]) throws {
        _ = try #require(model.beginEntry())
        try model.saveEntry(values: values, attested: false)
    }

    @Test func emptyHistoryPublishesAnExplicitRandomForecast() throws {
        let (model, _) = try readyModel()
        let forecast = try #require(model.card.forecast)
        #expect(forecast.mode == .random)
        #expect(forecast.values.count == 3)
        #expect(forecast.sum == forecast.values.reduce(0, +))
    }

    @Test func historyNeverFallsBackToRandom() async throws {
        let (model, _) = try readyModel()
        try addRoll(model, [6, 6, 1])
        let forecast = try #require(await publishedForecast(model))
        #expect(forecast.mode == .frequency)
        #expect(!forecast.hasEvidence)
        #expect(DiceRules.isValid(forecast.values, diceCount: 3))
        #expect(model.observationCount == 1)
    }

    @Test func unreadableStoreIsARecoveryStateNotAnEmptyDataset() {
        struct Broken: Error {}
        let model = AppModel(storeFactory: { throw Broken() })
        model.load()
        guard case .failed = model.loadState else {
            Issue.record("Expected a failed load state")
            return
        }
        #expect(model.round == nil)
        #expect(model.card.forecast == nil)
        #expect(model.beginEntry() == nil)
    }

    @Test func cancelingAPublishedForecastKeepsItUnchanged() throws {
        let (model, _) = try readyModel()
        let before = try #require(model.card.forecast)
        let draft = try #require(model.beginEntry())
        #expect(draft.predictionID == before.id)
        model.cancelEntry()
        #expect(model.card.forecast == before)
        #expect(model.observationCount == 0)
    }

    @Test func entryWithoutForecastStaysUnscoredAndCancelOpensAFreshTarget() async throws {
        let (model, _) = try readyModel(deadline: .seconds(60))
        try addRoll(model, [1, 2, 3])
        try addRoll(model, [2, 3, 4])
        // A third roll has lag context, so the forecast waits on bounded sequence fitting.
        _ = try #require(model.beginEntry())
        try model.saveEntry(values: [3, 4, 5], attested: false)
        #expect(model.card == .updating)

        let draft = try #require(model.beginEntry())
        #expect(draft.predictionID == nil)
        try await Task.sleep(for: .milliseconds(500))
        #expect(model.card.forecast == nil)

        model.cancelEntry()
        #expect(model.round?.targetID != draft.context.targetID)
        #expect(model.round?.expectedOrdinal == draft.context.expectedOrdinal)
        #expect(model.observationCount == 3)
    }

    @Test func sequenceForecastIsPublishedOnceTrainingFinishes() async throws {
        let (model, _) = try readyModel(deadline: .seconds(60))
        for index in 0..<6 {
            try addRoll(model, [index % 3 + 1, 1, 1])
        }
        let forecast = try #require(await publishedForecast(model))
        #expect(forecast.mode == .sequence)
        #expect(model.trainedPairs == 5)
    }

    @Test func clearObservationsReturnsToRandomOnlyAfterVerifiedEmptyHistory() throws {
        let (model, _) = try readyModel()
        try addRoll(model, [1, 1, 1])
        model.clearObservations()
        #expect(model.observationCount == 0)
        #expect(model.card.forecast?.mode == .random)
    }

    @Test func trainNewModelKeepsHistoryAndNeverUsesRandom() async throws {
        let (model, _) = try readyModel()
        try addRoll(model, [5, 5, 5])
        let run = model.dataset?.runID
        model.trainNewModel()
        #expect(model.dataset?.runID != run)
        #expect(model.observationCount == 1)
        #expect(await publishedForecast(model)?.mode == .frequency)
    }

    @Test func switchingTablesRestoresThePendingForecast() async throws {
        let (model, store) = try readyModel()
        try addRoll(model, [2, 2, 2])
        let original = try #require(await publishedForecast(model))
        let first = try #require(model.dataset?.id)
        model.createDataset(name: "Other", diceCount: 2)
        #expect(model.dataset?.diceCount == 2)
        model.selectDataset(first)
        #expect(model.card.forecast == original)
        _ = store
    }
}

@MainActor
struct MigrationTests {
    @Test func versionOneStoreMigratesWithItsHistory() throws {
        let directory = URL.temporaryDirectory.appending(path: "diceml-migration-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appending(path: "DiceML.store")

        // Write a store exactly as the first release did.
        let datasetID = UUID()
        let sessionID = UUID()
        do {
            let schema = Schema(versionedSchema: DiceSchemaV1.self)
            let container = try ModelContainer(
                for: schema,
                configurations: ModelConfiguration(schema: schema, url: url, cloudKitDatabase: .none)
            )
            let context = ModelContext(container)
            let dataset = DiceSchemaV1.DatasetRecord()
            dataset.id = datasetID
            dataset.name = "Table 1"
            dataset.diceCount = 3
            dataset.contextLength = 12
            dataset.generation = 1
            dataset.modelGeneration = 1
            dataset.nextOrdinal = 3
            dataset.currentSessionID = sessionID
            context.insert(dataset)
            for (ordinal, faces) in [(1, [1, 2, 3]), (2, [4, 5, 6])] {
                let roll = DiceSchemaV1.RollRecord()
                roll.datasetID = datasetID
                roll.datasetGeneration = 1
                roll.sessionID = sessionID
                roll.ordinal = ordinal
                roll.values = faces
                context.insert(roll)
            }
            let prediction = DiceSchemaV1.PredictionRecord()
            prediction.datasetID = datasetID
            prediction.datasetGeneration = 1
            prediction.mode = PredictionMode.frequency.rawValue
            context.insert(prediction)
            context.insert(DiceSchemaV1.ModelStateRecord())
            try context.save()
        }

        let store = try DiceStore.open(at: url)
        let dataset = try store.activeDataset()
        #expect(dataset.id == datasetID)
        #expect(try store.rolls(of: dataset).map(\.values) == [[1, 2, 3], [4, 5, 6]])
        #expect(try store.predictions(of: dataset).first?.snapshot.evidence == 0)
        #expect(try store.checkpoint(datasetID: dataset.id) == nil)
    }
}
