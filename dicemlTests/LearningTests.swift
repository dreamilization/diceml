import Foundation
import Testing
@testable import diceml

/// Deterministic generator for reproducible synthetic data.
struct SplitMix64: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

func makeRolls(_ values: [[Int]], session: UUID = UUID(), firstOrdinal: Int = 1) -> [RollValue] {
    values.enumerated().map { offset, faces in
        RollValue(id: UUID(), sessionID: session, ordinal: firstOrdinal + offset, values: faces)
    }
}

func identity() -> ModelIdentity {
    ModelIdentity(datasetID: UUID(), datasetGeneration: 1, modelGeneration: 1, runID: UUID())
}

func fairRolls(_ count: Int, dice: Int = 3, seed: UInt64) -> [RollValue] {
    var generator = SplitMix64(state: seed)
    return makeRolls((0..<count).map { _ in (0..<dice).map { _ in Int.random(in: 1...6, using: &generator) } })
}

/// Trains once on the full history and returns the predictor's forecast plus its walk-forward replay.
func trainedForecast(_ rolls: [RollValue], schema: FeatureSchema) throws -> ForecastResult {
    let checkpoint = try SequenceTrainer.run(TrainingJob(identity: identity(), schema: schema, rolls: rolls, base: nil))
    return Forecaster.learned(
        rolls: rolls,
        sessionID: rolls[0].sessionID,
        expectedOrdinal: (rolls.last?.ordinal ?? 0) + 1,
        schema: schema,
        checkpoint: checkpoint
    )
}

struct FeatureTests {
    @Test func segmentsBreakAtSessionChangesAndGaps() {
        let session = UUID()
        let rolls = makeRolls([[1], [2], [3]], session: session)
            + makeRolls([[4]], session: session, firstOrdinal: 5)
            + makeRolls([[5], [6]], session: UUID(), firstOrdinal: 6)
        let lengths = SequenceFeatures.segments(rolls).map(\.count)
        #expect(lengths == [3, 1, 2])
        #expect(SequenceFeatures.validPairCount(rolls) == 3)
        #expect(SequenceFeatures.trainingPairs(rolls, schema: FeatureSchema(diceCount: 1, contextLength: 2)).map(\.targetOrdinal) == [2, 3, 7])
    }

    @Test func featuresMaskMissingContextAndNeverIncludeTarget() {
        let schema = FeatureSchema(diceCount: 2, contextLength: 3)
        #expect(schema.inputDimension == 6 * 2 * 3 + 3)
        let rolls = makeRolls([[1, 6], [3, 4]])
        let features = SequenceFeatures.activeFeatures(lags: rolls[...], schema: schema)
        // Lag 0 is [3, 4]; lag 1 is [1, 6]; lag 2 is masked.
        #expect(features == [2, 6 + 3, 12 + 0, 12 + 6 + 5, 36 + 2])
    }

    @Test func currentLagsRequireTheNextOrdinalInTheSameSession() {
        let session = UUID()
        let rolls = makeRolls([[1], [2]], session: session)
        let schema = FeatureSchema(diceCount: 1, contextLength: 4)
        #expect(SequenceFeatures.currentLags(rolls, sessionID: session, expectedOrdinal: 3, schema: schema)?.count == 2)
        #expect(SequenceFeatures.currentLags(rolls, sessionID: session, expectedOrdinal: 4, schema: schema) == nil)
        #expect(SequenceFeatures.currentLags(rolls, sessionID: UUID(), expectedOrdinal: 3, schema: schema) == nil)
    }

    @Test func frequenciesAreSmoothed() {
        let q = FaceFrequencies.distributions(makeRolls([[3, 3]]), diceCount: 2)
        #expect(abs(q[0][2] - 2.0 / 7.0) < 1e-12)
        #expect(abs(q[1][0] - 1.0 / 7.0) < 1e-12)
    }

    @Test func argmaxBreaksTiesTowardTheLowestFace() {
        #expect(Forecaster.argmax([0.1, 0.3, 0.3, 0.1, 0.1, 0.1]) == 2)
        #expect(Forecaster.argmax(Array(repeating: 1.0 / 6.0, count: 6)) == 1)
    }

    @Test func fingerprintChangesWhenHistoryIsEdited() {
        let rolls = makeRolls([[1, 2], [3, 4]])
        let edited = [rolls[0], RollValue(id: rolls[1].id, sessionID: rolls[1].sessionID, ordinal: 2, values: [3, 5])]
        #expect(HistoryFingerprint.of(rolls, through: 1) == HistoryFingerprint.of(edited, through: 1))
        #expect(HistoryFingerprint.of(rolls, through: 2) != HistoryFingerprint.of(edited, through: 2))
    }
}

struct LearnerTests {
    @Test func learnsARepeatableSequence() throws {
        // Die 1 cycles 1...6; die 2 copies die 1's previous face.
        let rolls = makeRolls((0..<120).map { [$0 % 6 + 1, ($0 + 5) % 6 + 1] })
        let forecast = try trainedForecast(rolls, schema: FeatureSchema(diceCount: 2, contextLength: 2))

        #expect(forecast.mode == .sequence)
        #expect(forecast.values == [1, 6])
        #expect(forecast.probabilities.allSatisfy(Forecaster.isValidDistribution))
        #expect(forecast.blendWeight > 0.99)
        #expect(forecast.evidence > DiceRules.evidenceThreshold)
        let replay = try #require(forecast.replay)
        #expect(replay.modelLogLoss < replay.baselineLogLoss - 0.5)
    }

    @Test(arguments: [UInt64(1), 2, 3])
    func fairDiceGiveNoEvidenceAndNoSequenceWeight(seed: UInt64) throws {
        let forecast = try trainedForecast(fairRolls(150, seed: seed), schema: FeatureSchema(diceCount: 3, contextLength: 12))
        #expect(forecast.evidence < DiceRules.evidenceThreshold)
        #expect(forecast.blendWeight < 0.05)
        // The combined predictor stays near chance instead of becoming confidently wrong.
        let replay = try #require(forecast.replay)
        #expect(replay.modelLogLoss < EvaluationMetrics.uniformLogLoss + 0.02)
        #expect(replay.modelLogLoss <= replay.baselineLogLoss)
    }

    /// The walk-forward likelihood ratio is a test martingale under fair dice, so the share of fair
    /// histories that ever reach the threshold must stay near or below 5%.
    @Test func fairDiceRarelyEverReachLearned() {
        let schema = FeatureSchema(diceCount: 3, contextLength: 12)
        var crossed = 0
        for seed in 0..<100 {
            let rolls = fairRolls(120, seed: 1000 + UInt64(seed))
            let reached = stride(from: 4, through: rolls.count, by: 4).contains { count in
                Forecaster.learned(
                    rolls: Array(rolls[..<count]), sessionID: rolls[0].sessionID,
                    expectedOrdinal: count + 1, schema: schema, checkpoint: nil
                ).evidence >= DiceRules.evidenceThreshold
            }
            if reached { crossed += 1 }
        }
        #expect(crossed <= 7)
    }

    @Test func limitedHistoryIsLearnedButNotClaimedAsEvidence() throws {
        let rolls = fairRolls(6, seed: 9)
        let forecast = try trainedForecast(rolls, schema: FeatureSchema(diceCount: 3, contextLength: 12))
        #expect(forecast.mode != .random)
        #expect(forecast.evidence < DiceRules.evidenceThreshold)
        #expect(forecast.probabilities.allSatisfy { $0.max()! < 0.45 })
    }

    /// With one or two rolls every face is statistically tied, so the forecast must not simply echo
    /// the roll the user just entered.
    @Test func earlyForecastsDoNotEchoTheLatestRoll() {
        let schema = FeatureSchema(diceCount: 3, contextLength: 12)
        var echoedDice = 0
        var totalDice = 0
        for seed in 0..<200 {
            for count in [1, 2] {
                let rolls = fairRolls(count, seed: 5000 + UInt64(seed))
                let forecast = Forecaster.learned(
                    rolls: rolls, sessionID: rolls[0].sessionID, expectedOrdinal: count + 1,
                    schema: schema, checkpoint: nil)
                #expect(forecast.mode == .frequency)
                for (die, face) in forecast.values.enumerated() {
                    if face == rolls[count - 1].values[die] { echoedDice += 1 }
                    totalDice += 1
                }
            }
        }
        // Echoing would make this 100%; a neutral tie-break lands near 1 in 6.
        #expect(Double(echoedDice) / Double(totalDice) < 0.25)
    }

    @Test func forecastIsStableForTheSameHistory() {
        let rolls = fairRolls(5, seed: 42)
        let schema = FeatureSchema(diceCount: 3, contextLength: 12)
        let first = Forecaster.learned(rolls: rolls, sessionID: rolls[0].sessionID, expectedOrdinal: 6, schema: schema, checkpoint: nil)
        let second = Forecaster.learned(rolls: rolls, sessionID: rolls[0].sessionID, expectedOrdinal: 6, schema: schema, checkpoint: nil)
        #expect(first == second)
    }

    @Test func biasedDieIsDetectedByFrequencies() {
        var generator = SplitMix64(state: 77)
        let rolls = makeRolls((0..<200).map { _ in
            [Int.random(in: 0..<2, using: &generator) == 0 ? 6 : Int.random(in: 1...5, using: &generator),
             Int.random(in: 1...6, using: &generator)]
        })
        let forecast = Forecaster.learned(
            rolls: rolls, sessionID: rolls[0].sessionID, expectedOrdinal: 201,
            schema: FeatureSchema(diceCount: 2, contextLength: 4), checkpoint: nil)
        #expect(forecast.values[0] == 6)
        #expect(forecast.evidence > DiceRules.evidenceThreshold)
        #expect((forecast.replay?.modelLogLoss ?? .infinity) < EvaluationMetrics.uniformLogLoss)
    }

    @Test func warmStartDoesNotInflatePairCountOrRescoreHistory() throws {
        let schema = FeatureSchema(diceCount: 3, contextLength: 4)
        let rolls = fairRolls(45, seed: 11)
        let id = identity()
        let first = try SequenceTrainer.run(TrainingJob(identity: id, schema: schema, rolls: Array(rolls[..<40]), base: nil))
        #expect(first.trainedPairCount == 39)
        #expect(first.sequenceTargets == Array(2...40))

        let second = try SequenceTrainer.run(TrainingJob(identity: id, schema: schema, rolls: rolls, base: first))
        #expect(second.trainedPairCount == 44)
        #expect(second.sequenceTargets == Array(2...45))
        // Earlier walk-forward forecasts are preserved, never recomputed with later knowledge.
        #expect(Array(second.sequenceForecasts.prefix(first.sequenceForecasts.count)) == first.sequenceForecasts)

        let repeated = try SequenceTrainer.run(TrainingJob(identity: id, schema: schema, rolls: rolls, base: second))
        #expect(repeated == second)
    }

    @Test func checkpointFromAnotherRunIsNotReused() throws {
        let schema = FeatureSchema(diceCount: 1, contextLength: 2)
        let rolls = makeRolls((0..<20).map { [$0 % 6 + 1] })
        let old = try SequenceTrainer.run(TrainingJob(identity: identity(), schema: schema, rolls: rolls, base: nil))
        let fresh = try SequenceTrainer.run(TrainingJob(identity: identity(), schema: schema, rolls: rolls, base: old))
        let rebuilt = try SequenceTrainer.run(TrainingJob(identity: fresh.identity, schema: schema, rolls: rolls, base: nil))
        #expect(fresh == rebuilt)
    }

    @Test func staleCheckpointIsIgnoredByTheForecast() throws {
        let schema = FeatureSchema(diceCount: 1, contextLength: 2)
        let rolls = makeRolls((0..<30).map { [$0 % 3 + 1] })
        let checkpoint = try SequenceTrainer.run(TrainingJob(identity: identity(), schema: schema, rolls: Array(rolls[..<29]), base: nil))
        let forecast = Forecaster.learned(
            rolls: rolls, sessionID: rolls[0].sessionID, expectedOrdinal: 31, schema: schema, checkpoint: checkpoint)
        #expect(forecast.mode == .frequency)
        #expect(forecast.blendWeight == 0)
    }

    @Test func noLagContextUsesFrequenciesRegardlessOfTrainingVolume() throws {
        let schema = FeatureSchema(diceCount: 1, contextLength: 2)
        let rolls = makeRolls((0..<60).map { [$0 % 2 + 1] })
        let checkpoint = try SequenceTrainer.run(TrainingJob(identity: identity(), schema: schema, rolls: rolls, base: nil))
        let forecast = Forecaster.learned(rolls: rolls, sessionID: UUID(), expectedOrdinal: 61, schema: schema, checkpoint: checkpoint)
        #expect(forecast.mode == .frequency)
        #expect(forecast.blendWeight == 0)
    }

    @Test func evaluatorScoresAgainstFrozenBaselines() throws {
        let uniform = FaceFrequencies.uniform(diceCount: 2)
        let metrics = try #require(Evaluator.metrics([
            ScoredPrediction(values: [1, 2], probabilities: uniform, baseline: uniform, actual: [1, 3], timing: .preEntry),
        ]))
        #expect(metrics.modelDieHitRate == 0.5)
        #expect(metrics.modelExactRolls == 0)
        #expect(metrics.modelExactSums == 0)
        #expect(abs(metrics.modelLogLoss - EvaluationMetrics.uniformLogLoss) < 1e-12)
    }
}
