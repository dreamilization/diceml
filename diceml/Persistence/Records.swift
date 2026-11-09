import Foundation
import SwiftData

/// Observations from one explicitly compatible dice set and setup.
@Model
final class DatasetRecord {
    var id: UUID
    var name: String
    var diceCount: Int
    var orderingConvention: String
    var contextLength: Int
    var createdAt: Date
    var lastSelectedAt: Date
    /// Advanced by every clear; never reused, so late callbacks from cleared work are rejected.
    var generation: Int
    var modelGeneration: Int
    var activeRunID: UUID
    var historyRevision: Int
    var settingsRevision: Int
    /// Next roll ordinal. Monotonic: deleted ordinals leave gaps and are never reused.
    var nextOrdinal: Int
    var currentSessionID: UUID

    init(name: String, diceCount: Int, sessionID: UUID, now: Date = .now) {
        id = UUID()
        self.name = name
        self.diceCount = diceCount
        orderingConvention = DiceRules.orderingConvention
        contextLength = DiceRules.defaultContextLength
        createdAt = now
        lastSelectedAt = now
        generation = 1
        modelGeneration = 1
        activeRunID = UUID()
        historyRevision = 0
        settingsRevision = 0
        nextOrdinal = 1
        currentSessionID = sessionID
    }
}

/// A recording sequence. Its first roll never follows the last roll of another session.
@Model
final class SessionRecord {
    var id: UUID
    var datasetID: UUID
    var startedAt: Date
    var note: String?

    init(id: UUID = UUID(), datasetID: UUID, startedAt: Date = .now) {
        self.id = id
        self.datasetID = datasetID
        self.startedAt = startedAt
    }
}

/// An actual observed roll: the only source of training labels.
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

    init(
        submissionID: UUID,
        context: RoundContext,
        ordinal: Int,
        values: [Int],
        source: InputSource,
        observedAt: Date
    ) {
        id = UUID()
        self.submissionID = submissionID
        targetID = context.targetID
        datasetID = context.datasetID
        datasetGeneration = context.datasetGeneration
        sessionID = context.sessionID
        self.ordinal = ordinal
        self.values = values
        self.observedAt = observedAt
        recordedAt = .now
        self.source = source.rawValue
        revision = 0
    }

    var value: RollValue {
        RollValue(id: id, sessionID: sessionID, ordinal: ordinal, values: values)
    }
}

/// A published forecast frozen before its target was observed. Never used as a training label.
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
    /// Added in V2; optional so existing stores migrate. Older forecasts have no recorded evidence.
    var evidence: Double?
    var publishedAt: Date
    var timing: String?
    var matchedActualID: UUID?
    /// The target was abandoned before an actual roll arrived; it can never be scored.
    var retired: Bool
    /// A later edit or deletion changed this forecast's target or context.
    var revised: Bool

    init(_ snapshot: ForecastSnapshot) {
        id = snapshot.id
        datasetID = snapshot.context.datasetID
        datasetGeneration = snapshot.context.datasetGeneration
        modelGeneration = snapshot.context.modelGeneration
        runID = snapshot.context.runID
        sessionID = snapshot.context.sessionID
        targetID = snapshot.context.targetID
        expectedOrdinal = snapshot.context.expectedOrdinal
        historyRevision = snapshot.context.historyRevision
        settingsRevision = snapshot.context.settingsRevision
        featureSchemaVersion = DiceRules.featureSchemaVersion
        values = snapshot.values
        probabilityData = snapshot.probabilities.flattened.encodedData
        baselineData = snapshot.baseline.flattened.encodedData
        mode = snapshot.mode.rawValue
        trainedPairs = snapshot.trainedPairs
        blendWeight = snapshot.blendWeight
        evidence = snapshot.evidence
        publishedAt = snapshot.publishedAt
        retired = false
        revised = false
    }

    var context: RoundContext {
        RoundContext(
            datasetID: datasetID,
            datasetGeneration: datasetGeneration,
            modelGeneration: modelGeneration,
            runID: runID,
            sessionID: sessionID,
            targetID: targetID,
            expectedOrdinal: expectedOrdinal,
            historyRevision: historyRevision,
            settingsRevision: settingsRevision
        )
    }

    var probabilities: [[Double]] {
        [[Double]](unflattening: [Double](encodedData: probabilityData), rowLength: DiceRules.faceCount)
    }

    var baseline: [[Double]] {
        [[Double]](unflattening: [Double](encodedData: baselineData), rowLength: DiceRules.faceCount)
    }

    var snapshot: ForecastSnapshot {
        ForecastSnapshot(
            id: id,
            context: context,
            values: values,
            probabilities: probabilities,
            baseline: baseline,
            mode: PredictionMode(rawValue: mode) ?? .frequency,
            trainedPairs: trainedPairs,
            blendWeight: blendWeight,
            evidence: evidence ?? 0,
            publishedAt: publishedAt
        )
    }
}

/// Persisted sequence-model checkpoint. A recoverable cache that can be rebuilt from actual rolls.
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
    /// Added in V2; optional so existing stores migrate. A checkpoint without them is rebuilt.
    var forecastTargets: [Int]?
    @Attribute(.externalStorage) var forecastData: Data?
    var updatedAt: Date

    init(_ checkpoint: Checkpoint) {
        datasetID = checkpoint.identity.datasetID
        datasetGeneration = checkpoint.identity.datasetGeneration
        modelGeneration = checkpoint.identity.modelGeneration
        runID = checkpoint.identity.runID
        featureSchemaVersion = checkpoint.schema.version
        diceCount = checkpoint.schema.diceCount
        contextLength = checkpoint.schema.contextLength
        trainedThroughOrdinal = 0
        prefixHash = 0
        trainedPairCount = 0
        hyperparametersVersion = 0
        optimizerStep = 0
        weights = Data()
        firstMoment = Data()
        secondMoment = Data()
        updatedAt = .now
        update(from: checkpoint)
    }

    func update(from checkpoint: Checkpoint) {
        datasetGeneration = checkpoint.identity.datasetGeneration
        modelGeneration = checkpoint.identity.modelGeneration
        runID = checkpoint.identity.runID
        featureSchemaVersion = checkpoint.schema.version
        diceCount = checkpoint.schema.diceCount
        contextLength = checkpoint.schema.contextLength
        trainedThroughOrdinal = checkpoint.trainedThroughOrdinal
        prefixHash = Int64(bitPattern: checkpoint.prefixHash)
        trainedPairCount = checkpoint.trainedPairCount
        hyperparametersVersion = checkpoint.hyperparametersVersion
        optimizerStep = checkpoint.parameters.step
        weights = checkpoint.parameters.weights.encodedData
        firstMoment = checkpoint.parameters.firstMoment.encodedData
        secondMoment = checkpoint.parameters.secondMoment.encodedData
        forecastTargets = checkpoint.sequenceTargets
        forecastData = checkpoint.sequenceForecasts.encodedData
        updatedAt = .now
    }

    var checkpoint: Checkpoint {
        let schema = FeatureSchema(diceCount: diceCount, contextLength: contextLength, version: featureSchemaVersion)
        return Checkpoint(
            identity: ModelIdentity(
                datasetID: datasetID,
                datasetGeneration: datasetGeneration,
                modelGeneration: modelGeneration,
                runID: runID
            ),
            parameters: ModelParameters(
                schema: schema,
                weights: [Double](encodedData: weights),
                firstMoment: [Double](encodedData: firstMoment),
                secondMoment: [Double](encodedData: secondMoment),
                step: optimizerStep
            ),
            trainedThroughOrdinal: trainedThroughOrdinal,
            prefixHash: UInt64(bitPattern: prefixHash),
            trainedPairCount: trainedPairCount,
            hyperparametersVersion: hyperparametersVersion,
            sequenceTargets: forecastTargets ?? [],
            sequenceForecasts: forecastData.map { [Double](encodedData: $0) } ?? []
        )
    }
}

/// Current store layout. V2 adds forecast evidence to predictions and walk-forward sequence forecasts
/// to model checkpoints; both have defaults, so V1 stores migrate without custom code.
nonisolated enum DiceSchemaV2: VersionedSchema {
    static var versionIdentifier: Schema.Version { Schema.Version(2, 0, 0) }

    static var models: [any PersistentModel.Type] {
        [DatasetRecord.self, SessionRecord.self, RollRecord.self, PredictionRecord.self, ModelStateRecord.self]
    }
}

nonisolated enum DiceMigrationPlan: SchemaMigrationPlan {
    static var schemas: [any VersionedSchema.Type] { [DiceSchemaV1.self, DiceSchemaV2.self] }
    static var stages: [MigrationStage] {
        [.lightweight(fromVersion: DiceSchemaV1.self, toVersion: DiceSchemaV2.self)]
    }
}
