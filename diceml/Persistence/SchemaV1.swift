import Foundation
import SwiftData

/// The first shipped store layout, frozen so SwiftData can recognize and migrate existing stores.
/// Never edit these models; add a new schema version and a migration stage instead.
nonisolated enum DiceSchemaV1: VersionedSchema {
    static var versionIdentifier: Schema.Version { Schema.Version(1, 0, 0) }

    static var models: [any PersistentModel.Type] {
        [DatasetRecord.self, SessionRecord.self, RollRecord.self, PredictionRecord.self, ModelStateRecord.self]
    }

    @Model
    final class DatasetRecord {
        var id: UUID
        var name: String
        var diceCount: Int
        var orderingConvention: String
        var contextLength: Int
        var createdAt: Date
        var lastSelectedAt: Date
        var generation: Int
        var modelGeneration: Int
        var activeRunID: UUID
        var historyRevision: Int
        var settingsRevision: Int
        var nextOrdinal: Int
        var currentSessionID: UUID

        init() {
            id = UUID()
            name = ""
            diceCount = 0
            orderingConvention = ""
            contextLength = 0
            createdAt = .now
            lastSelectedAt = .now
            generation = 0
            modelGeneration = 0
            activeRunID = UUID()
            historyRevision = 0
            settingsRevision = 0
            nextOrdinal = 0
            currentSessionID = UUID()
        }
    }

    @Model
    final class SessionRecord {
        var id: UUID
        var datasetID: UUID
        var startedAt: Date
        var note: String?

        init() {
            id = UUID()
            datasetID = UUID()
            startedAt = .now
        }
    }

    @Model
    final class RollRecord {
        var id: UUID
        var submissionID: UUID
        var targetID: UUID
        var datasetID: UUID
        var datasetGeneration: Int
        var sessionID: UUID
        var ordinal: Int
        var values: [Int]
        var observedAt: Date
        var recordedAt: Date
        var source: String
        var revision: Int
        var editedAt: Date?

        init() {
            id = UUID()
            submissionID = UUID()
            targetID = UUID()
            datasetID = UUID()
            datasetGeneration = 0
            sessionID = UUID()
            ordinal = 0
            values = []
            observedAt = .now
            recordedAt = .now
            source = ""
            revision = 0
        }
    }

    @Model
    final class PredictionRecord {
        var id: UUID
        var datasetID: UUID
        var datasetGeneration: Int
        var modelGeneration: Int
        var runID: UUID
        var sessionID: UUID
        var targetID: UUID
        var expectedOrdinal: Int
        var historyRevision: Int
        var settingsRevision: Int
        var featureSchemaVersion: Int
        var values: [Int]
        var probabilityData: Data
        var baselineData: Data
        var mode: String
        var trainedPairs: Int
        var blendWeight: Double
        var publishedAt: Date
        var timing: String?
        var matchedActualID: UUID?
        var retired: Bool
        var revised: Bool

        init() {
            id = UUID()
            datasetID = UUID()
            datasetGeneration = 0
            modelGeneration = 0
            runID = UUID()
            sessionID = UUID()
            targetID = UUID()
            expectedOrdinal = 0
            historyRevision = 0
            settingsRevision = 0
            featureSchemaVersion = 0
            values = []
            probabilityData = Data()
            baselineData = Data()
            mode = ""
            trainedPairs = 0
            blendWeight = 0
            publishedAt = .now
            retired = false
            revised = false
        }
    }

    @Model
    final class ModelStateRecord {
        var datasetID: UUID
        var datasetGeneration: Int
        var modelGeneration: Int
        var runID: UUID
        var featureSchemaVersion: Int
        var diceCount: Int
        var contextLength: Int
        var trainedThroughOrdinal: Int
        var prefixHash: Int64
        var trainedPairCount: Int
        var hyperparametersVersion: Int
        var optimizerStep: Int
        @Attribute(.externalStorage) var weights: Data
        @Attribute(.externalStorage) var firstMoment: Data
        @Attribute(.externalStorage) var secondMoment: Data
        var updatedAt: Date

        init() {
            datasetID = UUID()
            datasetGeneration = 0
            modelGeneration = 0
            runID = UUID()
            featureSchemaVersion = 0
            diceCount = 0
            contextLength = 0
            trainedThroughOrdinal = 0
            prefixHash = 0
            trainedPairCount = 0
            hyperparametersVersion = 0
            optimizerStep = 0
            weights = Data()
            firstMoment = Data()
            secondMoment = Data()
            updatedAt = .now
        }
    }
}
