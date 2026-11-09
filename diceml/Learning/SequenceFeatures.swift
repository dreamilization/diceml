import Foundation

/// Shape of the lagged one-hot input. Different shapes require separate model parameters.
nonisolated struct FeatureSchema: Sendable, Hashable, Codable {
    let diceCount: Int
    let contextLength: Int
    var version: Int = DiceRules.featureSchemaVersion

    var isValid: Bool {
        DiceRules.diceCountRange.contains(diceCount) && DiceRules.contextLengthRange.contains(contextLength)
    }

    /// `6 * N * X` face indicators plus one missing-context mask per lag.
    var inputDimension: Int { DiceRules.faceCount * diceCount * contextLength + contextLength }
}

/// One next-roll training example built only from earlier actual rolls in the same segment.
nonisolated struct TrainingPair: Sendable {
    /// Indices of the input features whose value is one; every other feature is zero.
    let features: [Int]
    let targets: [Int]
    let targetOrdinal: Int
}

nonisolated enum SequenceFeatures {
    /// Splits history into contiguous segments. A segment ends at a session change or an ordinal gap,
    /// so deleted or missed rolls never make their surviving neighbors look adjacent.
    static func segments(_ rolls: [RollValue]) -> [[RollValue]] {
        var result: [[RollValue]] = []
        for roll in rolls.sorted(by: { $0.ordinal < $1.ordinal }) {
            if let last = result.last?.last, last.sessionID == roll.sessionID, last.ordinal + 1 == roll.ordinal {
                result[result.count - 1].append(roll)
            } else {
                result.append([roll])
            }
        }
        return result
    }

    /// Active feature indices for lagged rolls ordered oldest to newest. Lag 0 is the most recent roll.
    static func activeFeatures(lags: ArraySlice<RollValue>, schema: FeatureSchema) -> [Int] {
        let lagStride = DiceRules.faceCount * schema.diceCount
        let maskOffset = lagStride * schema.contextLength
        let recent = Array(lags.suffix(schema.contextLength))
        var indices: [Int] = []
        indices.reserveCapacity(schema.contextLength * schema.diceCount)
        for lag in 0..<schema.contextLength {
            let index = recent.count - 1 - lag
            if index >= 0 {
                for (die, face) in recent[index].values.prefix(schema.diceCount).enumerated() {
                    indices.append(lag * lagStride + die * DiceRules.faceCount + face - 1)
                }
            } else {
                indices.append(maskOffset + lag)
            }
        }
        return indices
    }

    /// All valid next-roll pairs in target-ordinal order. A segment's first roll is never a target.
    static func trainingPairs(_ rolls: [RollValue], schema: FeatureSchema) -> [TrainingPair] {
        var pairs: [TrainingPair] = []
        for segment in segments(rolls) where segment.count > 1 {
            for target in 1..<segment.count {
                let lags = segment[max(0, target - schema.contextLength)..<target]
                pairs.append(TrainingPair(
                    features: activeFeatures(lags: lags, schema: schema),
                    targets: segment[target].values,
                    targetOrdinal: segment[target].ordinal
                ))
            }
        }
        return pairs
    }

    /// `m = sum(max(segmentLength - 1, 0))`.
    static func validPairCount(_ rolls: [RollValue]) -> Int {
        segments(rolls).reduce(0) { $0 + max($1.count - 1, 0) }
    }

    /// Lag context for the next target, or `nil` when the target starts a new segment.
    static func currentLags(
        _ rolls: [RollValue],
        sessionID: UUID,
        expectedOrdinal: Int,
        schema: FeatureSchema
    ) -> ArraySlice<RollValue>? {
        guard let segment = segments(rolls).last,
              let latest = segment.last,
              latest.sessionID == sessionID,
              latest.ordinal + 1 == expectedOrdinal
        else { return nil }
        return segment.suffix(schema.contextLength)
    }
}

nonisolated enum FaceFrequencies {
    /// Smoothed per-position face frequencies `q(k) = (count(k) + 1) / (T + 6)`.
    static func distributions(_ rolls: [RollValue], diceCount: Int) -> [[Double]] {
        var counts = Array(repeating: Array(repeating: 0.0, count: DiceRules.faceCount), count: diceCount)
        for roll in rolls {
            for (die, face) in roll.values.prefix(diceCount).enumerated() where DiceRules.faces.contains(face) {
                counts[die][face - 1] += 1
            }
        }
        let total = Double(rolls.count)
        return counts.map { row in row.map { ($0 + 1) / (total + Double(DiceRules.faceCount)) } }
    }

    static func uniform(diceCount: Int) -> [[Double]] {
        Array(repeating: Array(repeating: 1 / Double(DiceRules.faceCount), count: DiceRules.faceCount), count: diceCount)
    }
}

/// Stable FNV-1a fingerprint of a history prefix. Swift's `Hasher` is seeded per launch, so it cannot be persisted.
nonisolated enum HistoryFingerprint {
    static func of(_ rolls: [RollValue], through ordinal: Int) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        func mix(_ byte: UInt8) {
            hash ^= UInt64(byte)
            hash &*= 0x0000_0100_0000_01b3
        }
        for roll in rolls.sorted(by: { $0.ordinal < $1.ordinal }) where roll.ordinal <= ordinal {
            withUnsafeBytes(of: Int64(roll.ordinal).littleEndian) { $0.forEach(mix) }
            withUnsafeBytes(of: roll.sessionID.uuid) { $0.forEach(mix) }
            roll.values.forEach { mix(UInt8(clamping: $0)) }
            mix(0xFF)
        }
        return hash
    }
}
