import Foundation

/// The dataset and training run a checkpoint belongs to.
nonisolated struct ModelIdentity: Sendable, Hashable {
    let datasetID: UUID
    let datasetGeneration: Int
    let modelGeneration: Int
    let runID: UUID
}

/// A derived, rebuildable model cache together with the history prefix it was trained on.
nonisolated struct Checkpoint: Sendable, Equatable {
    let identity: ModelIdentity
    var parameters: ModelParameters
    /// Highest roll ordinal included in training, or `0` before any roll.
    var trainedThroughOrdinal: Int
    var prefixHash: UInt64
    /// Unique next-roll pairs learned; rehearsal never increases it.
    var trainedPairCount: Int
    var hyperparametersVersion: Int
    /// Target ordinal of every learned pair, in training order.
    var sequenceTargets: [Int] = []
    /// Per-die distributions (flattened `pairs × N × 6`) the model gave each pair *before* training on it.
    /// These walk-forward forecasts decide how much the sequence model is trusted.
    var sequenceForecasts: [Double] = []

    var schema: FeatureSchema { parameters.schema }

    /// True when the history this checkpoint was trained on is still an unchanged prefix of `rolls`.
    func isValidPrefix(of rolls: [RollValue]) -> Bool {
        prefixHash == HistoryFingerprint.of(rolls, through: trainedThroughOrdinal)
    }

    /// True when the checkpoint has learned exactly the current history.
    func covers(_ rolls: [RollValue]) -> Bool {
        trainedThroughOrdinal == (rolls.map(\.ordinal).max() ?? 0) && isValidPrefix(of: rolls)
    }
}

/// An immutable snapshot of everything the trainer needs. Never contains live SwiftData objects.
nonisolated struct TrainingJob: Sendable {
    let identity: ModelIdentity
    let schema: FeatureSchema
    let rolls: [RollValue]
    let base: Checkpoint?
    var hyperparameters: TrainingHyperparameters = .standard
}

/// Fits the sequence model off the main actor. Work is cancelable between optimizer steps.
actor SequenceTrainer {
    func train(_ job: TrainingJob) throws -> Checkpoint {
        try Self.run(job)
    }

    nonisolated static func run(_ job: TrainingJob) throws -> Checkpoint {
        let hyper = job.hyperparameters
        let pairs = SequenceFeatures.trainingPairs(job.rolls, schema: job.schema)
        var parameters = ModelParameters.zeros(job.schema)
        var trainedThrough = 0
        var learned: [TrainingPair] = []
        var targets: [Int] = []
        var forecasts: [Double] = []
        let width = job.schema.diceCount * DiceRules.faceCount

        // Warm-start only from an unchanged prefix of the same run, schema, and schedule,
        // with a complete walk-forward record for every pair it learned.
        if let base = job.base,
           base.identity == job.identity,
           base.schema == job.schema,
           base.hyperparametersVersion == hyper.version,
           base.parameters.isConsistent,
           base.isValidPrefix(of: job.rolls) {
            let prior = pairs.filter { $0.targetOrdinal <= base.trainedThroughOrdinal }
            if base.sequenceTargets == prior.map(\.targetOrdinal),
               base.sequenceForecasts.count == prior.count * width {
                parameters = base.parameters
                trainedThrough = base.trainedThroughOrdinal
                learned = prior
                targets = base.sequenceTargets
                forecasts = base.sequenceForecasts
            }
        }

        let fresh = pairs.filter { $0.targetOrdinal > trainedThrough }
        var start = 0
        while start < fresh.count {
            try Task.checkCancellation()
            let chunk = Array(fresh[start..<min(start + hyper.chunkSize, fresh.count)])
            // Record out-of-sample forecasts before the chunk can influence the parameters.
            for pair in chunk {
                forecasts += LogisticSequenceModel.probabilities(parameters, features: pair.features).flattened
                targets.append(pair.targetOrdinal)
            }
            let rehearsal = rehearsalBatch(from: learned, limit: hyper.rehearsalLimit, salt: parameters.step)
            try LogisticSequenceModel.fit(&parameters, batch: chunk + rehearsal, hyperparameters: hyper)
            learned.append(contentsOf: chunk)
            start += hyper.chunkSize
        }

        let latest = job.rolls.map(\.ordinal).max() ?? 0
        return Checkpoint(
            identity: job.identity,
            parameters: parameters,
            trainedThroughOrdinal: latest,
            prefixHash: HistoryFingerprint.of(job.rolls, through: latest),
            trainedPairCount: learned.count,
            hyperparametersVersion: hyper.version,
            sequenceTargets: targets,
            sequenceForecasts: forecasts
        )
    }

    /// Deterministic, bounded strided sample of already-learned pairs.
    nonisolated static func rehearsalBatch(from learned: [TrainingPair], limit: Int, salt: Int) -> [TrainingPair] {
        guard learned.count > limit, limit > 0 else { return learned }
        let stride = learned.count / limit
        let offset = salt % stride
        return (0..<limit).map { learned[offset + $0 * stride] }
    }
}
