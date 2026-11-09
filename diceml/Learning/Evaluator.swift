import Foundation

/// An immutable pre-target forecast joined with the actual roll it was scored against.
nonisolated struct ScoredPrediction: Sendable {
    let values: [Int]
    let probabilities: [[Double]]
    let baseline: [[Double]]
    let actual: [Int]
    let timing: TimingCategory
}

nonisolated struct EvaluationMetrics: Sendable, Equatable {
    let count: Int
    let diceCount: Int
    let modelLogLoss: Double
    let baselineLogLoss: Double
    let modelDieHitRate: Double
    let baselineDieHitRate: Double
    let modelExactRolls: Int
    let baselineExactRolls: Int
    let modelExactSums: Int
    let baselineExactSums: Int

    static let uniformLogLoss = Foundation.log(Double(DiceRules.faceCount))

    /// Rough binomial standard error of a per-die hit rate.
    func standardError(of rate: Double) -> Double {
        let n = Double(count * diceCount)
        return n > 0 ? (rate * (1 - rate) / n).squareRoot() : 0
    }
}

/// Walk-forward scoring of published forecasts against later actual rolls, per timing category.
nonisolated enum Evaluator {
    static func evaluate(_ predictions: [ScoredPrediction]) -> [TimingCategory: EvaluationMetrics] {
        Dictionary(grouping: predictions, by: \.timing).compactMapValues(metrics)
    }

    static func metrics(_ predictions: [ScoredPrediction]) -> EvaluationMetrics? {
        guard let diceCount = predictions.first?.actual.count, diceCount > 0 else { return nil }
        var modelLoss = 0.0, baselineLoss = 0.0
        var modelHits = 0, baselineHits = 0
        var modelExact = 0, baselineExact = 0, modelSums = 0, baselineSums = 0
        var dieCount = 0

        for prediction in predictions {
            let baselineValues = prediction.baseline.map(Forecaster.argmax)
            let actualSum = prediction.actual.reduce(0, +)
            for (die, face) in prediction.actual.enumerated() {
                modelLoss -= Foundation.log(max(prediction.probabilities[die][face - 1], 1e-15))
                baselineLoss -= Foundation.log(max(prediction.baseline[die][face - 1], 1e-15))
                if prediction.values[die] == face { modelHits += 1 }
                if baselineValues[die] == face { baselineHits += 1 }
                dieCount += 1
            }
            if prediction.values == prediction.actual { modelExact += 1 }
            if baselineValues == prediction.actual { baselineExact += 1 }
            if prediction.values.reduce(0, +) == actualSum { modelSums += 1 }
            if baselineValues.reduce(0, +) == actualSum { baselineSums += 1 }
        }

        let dice = Double(dieCount)
        return EvaluationMetrics(
            count: predictions.count,
            diceCount: diceCount,
            modelLogLoss: modelLoss / dice,
            baselineLogLoss: baselineLoss / dice,
            modelDieHitRate: Double(modelHits) / dice,
            baselineDieHitRate: Double(baselineHits) / dice,
            modelExactRolls: modelExact,
            baselineExactRolls: baselineExact,
            modelExactSums: modelSums,
            baselineExactSums: baselineSums
        )
    }
}
