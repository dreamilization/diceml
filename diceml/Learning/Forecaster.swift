import Foundation

nonisolated struct ForecastResult: Sendable, Equatable {
    let values: [Int]
    let probabilities: [[Double]]
    /// Plain smoothed per-position frequencies, frozen as the evaluation baseline.
    let baseline: [[Double]]
    let mode: PredictionMode
    let trainedPairs: Int
    /// Evidence-based weight of the sequence model in this forecast; zero when it was not used.
    let blendWeight: Double
    /// Walk-forward log likelihood ratio, in nats, of this predictor against fair independent dice.
    let evidence: Double
    /// Walk-forward replay of this predictor over the existing history.
    let replay: EvaluationMetrics?
}

/// Combines learned face frequencies and the sequence model by how well each actually predicted
/// earlier rolls before seeing them (prequential Bayesian model averaging).
///
/// With little data every component still predicts close to chance, so the combined forecast stays
/// close to chance and its evidence stays low. Nothing is trusted because of training volume alone.
nonisolated enum Forecaster {
    /// One frequency component: per-position counts shrunk toward the dice-wide pooled distribution
    /// with strength `alpha`, which is itself shrunk toward uniform with strength `beta`.
    /// Infinite strengths mean "identical to the parent distribution".
    struct FrequencyComponent: Sendable, Equatable {
        let alpha: Double
        let beta: Double
    }

    static let frequencyComponents: [FrequencyComponent] = {
        let strengths: [Double] = [1, 6, 36, 216, .infinity]
        return strengths.flatMap { alpha in
            [6.0, 36, 216, 1296, .infinity].map { FrequencyComponent(alpha: alpha, beta: $0) }
        }
    }()

    /// Prior weight of the sequence model against frequencies before any walk-forward evidence.
    static let sequencePrior = 0.1

    /// Occam prior over frequency components: half the mass on "fair dice" (both strengths infinite),
    /// the rest shared equally. With little data the forecast therefore stays close to chance instead of
    /// over-trusting a handful of rolls, while real bias still accumulates weight quickly.
    static let logFrequencyPriors: [Double] = {
        let fairShare = 0.5
        let others = Double(frequencyComponents.count - 1)
        return frequencyComponents.map { component in
            component.alpha.isInfinite && component.beta.isInfinite
                ? Foundation.log(fairShare)
                : Foundation.log((1 - fairShare) / others)
        }
    }()

    /// Cold start for a successfully loaded, empty dataset only.
    static func random<G: RandomNumberGenerator>(diceCount: Int, using generator: inout G) -> ForecastResult {
        let uniform = FaceFrequencies.uniform(diceCount: diceCount)
        return ForecastResult(
            values: (0..<diceCount).map { _ in Int.random(in: DiceRules.faces, using: &generator) },
            probabilities: uniform,
            baseline: uniform,
            mode: .random,
            trainedPairs: 0,
            blendWeight: 0,
            evidence: 0,
            replay: nil
        )
    }

    /// Learned forecast for the next roll. `checkpoint` contributes only when it was trained on
    /// exactly the current history; otherwise the forecast uses learned frequencies alone.
    static func learned(
        rolls: [RollValue],
        sessionID: UUID,
        expectedOrdinal: Int,
        schema: FeatureSchema,
        checkpoint: Checkpoint?
    ) -> ForecastResult {
        let diceCount = schema.diceCount
        let width = diceCount * DiceRules.faceCount
        let sorted = rolls
            .filter { DiceRules.isValid($0.values, diceCount: diceCount) }
            .sorted { $0.ordinal < $1.ordinal }
        let sequence = checkpoint.flatMap { usableSequence($0, rolls: sorted, schema: schema) }

        var frequencies = FrequencyState(diceCount: diceCount)
        var logComponentWeights = logFrequencyPriors
        var logFrequencyEvidence = Foundation.log(1 - sequencePrior)
        var logSequenceEvidence = Foundation.log(sequencePrior)
        let logUniformRoll = -Double(diceCount) * Foundation.log(Double(DiceRules.faceCount))
        var evidence = 0.0
        var replay: [ScoredPrediction] = []
        var historySeed = HistorySeed()

        for roll in sorted {
            // Everything below uses only rolls before this one.
            let (frequencyMarginals, componentLogProbabilities) = frequencies.predict(
                weights: logComponentWeights, actual: roll.values)
            let frequencyLogProbability = logSumExp(zip(logComponentWeights, componentLogProbabilities).map { $0 + $1 })

            var marginals = frequencyMarginals
            var logProbability = frequencyLogProbability
            if let sequence, let index = sequence.index[roll.ordinal] {
                let sequenceMarginals = Array(sequence.forecasts[(index * width)..<((index + 1) * width)])
                let sequenceLogProbability = jointLogProbability(sequenceMarginals, actual: roll.values)
                let weight = sequenceWeight(logFrequencyEvidence, logSequenceEvidence)
                marginals = zip(frequencyMarginals, sequenceMarginals).map { (1 - weight) * $0 + weight * $1 }
                logProbability = logSumExp([
                    Foundation.log(1 - weight) + frequencyLogProbability,
                    Foundation.log(weight) + sequenceLogProbability,
                ])
                logFrequencyEvidence += frequencyLogProbability
                logSequenceEvidence += sequenceLogProbability
            }

            if frequencies.rollCount > 0 {
                evidence += logProbability - logUniformRoll
                replay.append(ScoredPrediction(
                    values: selectFaces(
                        rows(marginals, diceCount: diceCount),
                        observations: frequencies.rollCount,
                        seed: historySeed.value(for: roll.ordinal)
                    ),
                    probabilities: rows(marginals, diceCount: diceCount),
                    baseline: frequencies.laplace(),
                    actual: roll.values,
                    timing: .retrospective
                ))
            }

            logComponentWeights = normalizedLog(zip(logComponentWeights, componentLogProbabilities).map { $0 + $1 })
            frequencies.observe(roll.values)
            historySeed.observe(roll)
        }

        let frequencyNext = frequencies.predict(weights: logComponentWeights, actual: nil).marginals
        var probabilities = frequencyNext
        var mode = PredictionMode.frequency
        var weight = 0.0
        if let sequence,
           let lags = SequenceFeatures.currentLags(
               sorted, sessionID: sessionID, expectedOrdinal: expectedOrdinal, schema: schema) {
            let sequenceNext = LogisticSequenceModel.probabilities(
                sequence.checkpoint.parameters,
                features: SequenceFeatures.activeFeatures(lags: lags, schema: schema)
            ).flattened
            weight = sequenceWeight(logFrequencyEvidence, logSequenceEvidence)
            probabilities = zip(frequencyNext, sequenceNext).map { (1 - weight) * $0 + weight * $1 }
            mode = .sequence
        }

        var distributions = rows(probabilities, diceCount: diceCount)
        if !distributions.allSatisfy(isValidDistribution) {
            distributions = rows(frequencyNext, diceCount: diceCount)
            mode = .frequency
            weight = 0
        }

        return ForecastResult(
            values: selectFaces(
                distributions,
                observations: frequencies.rollCount,
                seed: historySeed.value(for: expectedOrdinal)
            ),
            probabilities: distributions,
            baseline: frequencies.laplace(),
            mode: mode,
            trainedPairs: sequence?.checkpoint.trainedPairCount ?? 0,
            blendWeight: weight,
            evidence: evidence.isFinite ? evidence : -.infinity,
            replay: Evaluator.metrics(replay)
        )
    }

    /// Most probable face, breaking exact ties by the lowest face value.
    static func argmax(_ distribution: [Double]) -> Int {
        var best = 0
        for index in distribution.indices where distribution[index] > distribution[best] {
            best = index
        }
        return best + 1
    }

    static func isValidDistribution(_ distribution: [Double]) -> Bool {
        distribution.count == DiceRules.faceCount
            && distribution.allSatisfy { $0.isFinite && $0 >= 0 }
            && abs(distribution.reduce(0, +) - 1) < 1e-9
    }

    // MARK: Helpers

    private struct SequenceForecasts {
        let checkpoint: Checkpoint
        let forecasts: [Double]
        let index: [Int: Int]
    }

    /// The checkpoint's stored walk-forward forecasts, if it covers exactly this history.
    private static func usableSequence(_ checkpoint: Checkpoint, rolls: [RollValue], schema: FeatureSchema) -> SequenceForecasts? {
        let width = schema.diceCount * DiceRules.faceCount
        guard checkpoint.schema == schema,
              checkpoint.trainedPairCount > 0,
              checkpoint.trainedPairCount == checkpoint.sequenceTargets.count,
              checkpoint.sequenceForecasts.count == checkpoint.sequenceTargets.count * width,
              checkpoint.sequenceForecasts.allSatisfy({ $0.isFinite && $0 >= 0 }),
              checkpoint.covers(rolls)
        else { return nil }
        var index: [Int: Int] = [:]
        for (position, target) in checkpoint.sequenceTargets.enumerated() { index[target] = position }
        return SequenceForecasts(checkpoint: checkpoint, forecasts: checkpoint.sequenceForecasts, index: index)
    }

    private static func sequenceWeight(_ logFrequency: Double, _ logSequence: Double) -> Double {
        let weight = 1 / (1 + Foundation.exp(logFrequency - logSequence))
        return weight.isFinite ? min(max(weight, 0), 1) : 0
    }

    private static func jointLogProbability(_ marginals: [Double], actual: [Int]) -> Double {
        actual.enumerated().reduce(0) { total, item in
            total + Foundation.log(max(marginals[item.offset * DiceRules.faceCount + item.element - 1], 1e-300))
        }
    }

    private static func rows(_ flat: [Double], diceCount: Int) -> [[Double]] {
        [[Double]](unflattening: flat, rowLength: DiceRules.faceCount)
    }

    /// Chooses the displayed face for each die.
    ///
    /// Faces whose probability is within sampling noise of the best face — `sqrt(p(1 - p) / (n + 1))`
    /// after `n` recorded rolls — are statistically indistinguishable, so they are treated as tied.
    /// Ties are broken by a deterministic value derived from the history and target, not by which
    /// face happened to appear last. With little data almost every face is tied, so the forecast no
    /// longer simply echoes the latest roll; a genuine bias or pattern still clears the band and wins.
    static func selectFaces(_ distributions: [[Double]], observations: Int, seed: UInt64) -> [Int] {
        distributions.enumerated().map { die, distribution in
            let best = distribution.max() ?? 0
            let tolerance = (best * (1 - best) / Double(observations + 1)).squareRoot()
            let tied = distribution.indices.filter { distribution[$0] >= best - tolerance }
            guard tied.count > 1 else { return argmax(distribution) }
            var mixed = seed ^ (UInt64(die + 1) &* 0x9E37_79B9_7F4A_7C15)
            mixed = (mixed ^ (mixed >> 31)) &* 0xBF58_476D_1CE4_E5B9
            mixed ^= mixed >> 29
            return tied[Int(mixed % UInt64(tied.count))] + 1
        }
    }

    /// Running FNV-1a hash of the recorded rolls, so the same history always yields the same tie-break.
    private struct HistorySeed {
        private var hash: UInt64 = 0xcbf2_9ce4_8422_2325

        mutating func observe(_ roll: RollValue) {
            mix(UInt64(bitPattern: Int64(roll.ordinal)))
            roll.values.forEach { mix(UInt64($0)) }
        }

        func value(for ordinal: Int) -> UInt64 {
            var copy = self
            copy.mix(UInt64(bitPattern: Int64(ordinal)))
            return copy.hash
        }

        private mutating func mix(_ value: UInt64) {
            for shift in stride(from: 0, to: 64, by: 8) {
                hash ^= (value >> UInt64(shift)) & 0xFF
                hash &*= 0x0000_0100_0000_01b3
            }
        }
    }

    static func logSumExp(_ values: [Double]) -> Double {
        guard let maximum = values.max(), maximum.isFinite else { return values.max() ?? -.infinity }
        return maximum + Foundation.log(values.reduce(0) { $0 + Foundation.exp($1 - maximum) })
    }

    private static func normalizedLog(_ values: [Double]) -> [Double] {
        let total = logSumExp(values)
        return values.map { $0 - total }
    }

    /// Running counts behind the frequency components.
    private struct FrequencyState {
        let diceCount: Int
        var counts: [Double]
        var pooled: [Double]
        var rollCount = 0

        init(diceCount: Int) {
            self.diceCount = diceCount
            counts = [Double](repeating: 0, count: diceCount * DiceRules.faceCount)
            pooled = [Double](repeating: 0, count: DiceRules.faceCount)
        }

        mutating func observe(_ values: [Int]) {
            for (die, face) in values.enumerated() {
                counts[die * DiceRules.faceCount + face - 1] += 1
                pooled[face - 1] += 1
            }
            rollCount += 1
        }

        /// Plain per-position smoothing `(count + 1) / (T + 6)`: the design's frequency baseline.
        func laplace() -> [[Double]] {
            let total = Double(rollCount) + Double(DiceRules.faceCount)
            return (0..<diceCount).map { die in
                (0..<DiceRules.faceCount).map { (counts[die * DiceRules.faceCount + $0] + 1) / total }
            }
        }

        /// Mixture marginals and, when `actual` is given, each component's joint log probability of it.
        func predict(weights logWeights: [Double], actual: [Int]?) -> (marginals: [Double], componentLogProbabilities: [Double]) {
            let faces = DiceRules.faceCount
            let n = Double(rollCount)
            let pooledTotal = n * Double(diceCount)
            let maximum = logWeights.max() ?? 0
            let weights = logWeights.map { Foundation.exp($0 - maximum) }
            let weightTotal = weights.reduce(0, +)
            var marginals = [Double](repeating: 0, count: diceCount * faces)
            var logProbabilities = [Double](repeating: 0, count: Forecaster.frequencyComponents.count)
            var parent = [Double](repeating: 0, count: faces)

            for (index, component) in Forecaster.frequencyComponents.enumerated() {
                for face in 0..<faces {
                    parent[face] = component.beta.isInfinite
                        ? 1 / Double(faces)
                        : (pooled[face] + component.beta / Double(faces)) / (pooledTotal + component.beta)
                }
                let share = weights[index] / weightTotal
                var logProbability = 0.0
                for die in 0..<diceCount {
                    for face in 0..<faces {
                        let probability = component.alpha.isInfinite
                            ? parent[face]
                            : (counts[die * faces + face] + component.alpha * parent[face]) / (n + component.alpha)
                        marginals[die * faces + face] += share * probability
                        if let actual, actual[die] == face + 1 {
                            logProbability += Foundation.log(probability)
                        }
                    }
                }
                logProbabilities[index] = logProbability
            }
            return (marginals, logProbabilities)
        }
    }
}
