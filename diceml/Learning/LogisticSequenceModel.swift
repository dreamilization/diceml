import Foundation

/// Weights and Adam optimizer state for one six-class softmax head per die.
/// Each head sees every lagged die position. Layout: `[die][face][feature..., bias]`.
nonisolated struct ModelParameters: Sendable, Equatable {
    let schema: FeatureSchema
    var weights: [Double]
    var firstMoment: [Double]
    var secondMoment: [Double]
    var step: Int

    var rowLength: Int { schema.inputDimension + 1 }

    static func zeros(_ schema: FeatureSchema) -> ModelParameters {
        let count = schema.diceCount * DiceRules.faceCount * (schema.inputDimension + 1)
        let zeros = [Double](repeating: 0, count: count)
        return ModelParameters(schema: schema, weights: zeros, firstMoment: zeros, secondMoment: zeros, step: 0)
    }

    var isConsistent: Bool {
        let count = schema.diceCount * DiceRules.faceCount * rowLength
        return weights.count == count && firstMoment.count == count && secondMoment.count == count
            && weights.allSatisfy(\.isFinite)
    }
}

/// A deterministic optimization schedule. Changing any value requires a new `version`.
nonisolated struct TrainingHyperparameters: Sendable, Equatable {
    var learningRate = 0.05
    /// Chosen by walk-forward comparison on synthetic fair, biased, cyclic, and sticky sequences:
    /// weaker penalties overfit fair dice well past uniform log loss.
    var l2 = 0.2
    var stepsPerChunk = 20
    var chunkSize = 32
    var rehearsalLimit = 96
    var version = 3

    static let standard = TrainingHyperparameters()
}

/// Regularized multinomial logistic regression over lagged one-hot rolls.
nonisolated enum LogisticSequenceModel {
    static func probabilities(_ parameters: ModelParameters, features: [Int]) -> [[Double]] {
        let dimension = parameters.schema.inputDimension
        let rowLength = parameters.rowLength
        return parameters.weights.withUnsafeBufferPointer { weights in
            (0..<parameters.schema.diceCount).map { die in
                let logits = (0..<DiceRules.faceCount).map { face in
                    let base = (die * DiceRules.faceCount + face) * rowLength
                    var logit = weights[base + dimension]
                    for feature in features { logit += weights[base + feature] }
                    return logit
                }
                return softmax(logits)
            }
        }
    }

    /// Numerically stable softmax.
    static func softmax(_ logits: [Double]) -> [Double] {
        let maximum = logits.max() ?? 0
        let exponentials = logits.map { Foundation.exp($0 - maximum) }
        let total = exponentials.reduce(0, +)
        return exponentials.map { $0 / total }
    }

    /// Runs a fixed number of full-batch Adam steps on mean cross-entropy plus L2 (bias excluded).
    static func fit(
        _ parameters: inout ModelParameters,
        batch: [TrainingPair],
        hyperparameters: TrainingHyperparameters
    ) throws {
        guard !batch.isEmpty else { return }
        let diceCount = parameters.schema.diceCount
        let dimension = parameters.schema.inputDimension
        let rowLength = parameters.rowLength
        let count = parameters.weights.count
        let scale = 1 / Double(batch.count)
        let beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8
        var gradient = [Double](repeating: 0, count: count)

        for _ in 0..<hyperparameters.stepsPerChunk {
            try Task.checkCancellation()
            gradient.withUnsafeMutableBufferPointer { $0.update(repeating: 0) }

            for pair in batch {
                let predicted = probabilities(parameters, features: pair.features)
                for die in 0..<diceCount {
                    let target = pair.targets[die] - 1
                    for face in 0..<DiceRules.faceCount {
                        let error = (predicted[die][face] - (face == target ? 1 : 0)) * scale
                        let base = (die * DiceRules.faceCount + face) * rowLength
                        gradient[base + dimension] += error
                        for feature in pair.features { gradient[base + feature] += error }
                    }
                }
            }

            parameters.step += 1
            let correction1 = 1 - Foundation.pow(beta1, Double(parameters.step))
            let correction2 = 1 - Foundation.pow(beta2, Double(parameters.step))
            let rate = hyperparameters.learningRate
            let l2 = hyperparameters.l2
            parameters.weights.withUnsafeMutableBufferPointer { weights in
                parameters.firstMoment.withUnsafeMutableBufferPointer { first in
                    parameters.secondMoment.withUnsafeMutableBufferPointer { second in
                        for index in 0..<count {
                            var g = gradient[index]
                            if index % rowLength != dimension { g += l2 * weights[index] }
                            first[index] = beta1 * first[index] + (1 - beta1) * g
                            second[index] = beta2 * second[index] + (1 - beta2) * g * g
                            let mHat = first[index] / correction1
                            let vHat = second[index] / correction2
                            weights[index] -= rate * mHat / (vHat.squareRoot() + epsilon)
                        }
                    }
                }
            }
        }
    }
}
