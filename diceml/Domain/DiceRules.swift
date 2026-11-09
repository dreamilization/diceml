import Foundation

/// Fixed rules and proposed version 1.0 bounds shared by storage, learning, and the interface.
nonisolated enum DiceRules {
    static let faces = 1...6
    static let faceCount = 6
    static let diceCountRange = 1...12
    static let defaultDiceCount = 3
    static let contextLengthRange = 1...64
    static let defaultContextLength = 12
    static let featureSchemaVersion = 1
    /// Pseudo-count in the shrinkage weight `w = m / (m + prior)`; a starting hyperparameter to validate.
    /// Walk-forward likelihood ratio against fair independent dice required before a forecast is labeled
    /// learned. The ratio is a nonnegative martingale under fair dice, so by Ville's inequality the chance
    /// it *ever* reaches 20 is at most 5%, however long the user keeps recording.
    static let evidenceThreshold = Foundation.log(20.0)
    /// Positional slot convention used by manual and camera entry within a dataset.
    static let orderingConvention = "left-to-right/top-to-bottom"

    static func isValid(_ values: [Int], diceCount: Int) -> Bool {
        values.count == diceCount && values.allSatisfy { faces.contains($0) }
    }
}

/// One actual observed roll, as an immutable value safe to send to the trainer.
nonisolated struct RollValue: Sendable, Hashable {
    let id: UUID
    let sessionID: UUID
    let ordinal: Int
    let values: [Int]
}

nonisolated enum PredictionMode: String, Sendable, Codable {
    case random
    case frequency
    case sequence
}

nonisolated enum TimingCategory: String, Sendable, Codable, CaseIterable {
    case retrospective
    case preEntry
    case attested
}

nonisolated enum InputSource: String, Sendable, Codable {
    case manual
    case camera
}

/// Identifies one pending prediction target. Any change to these fields means a different target.
nonisolated struct RoundContext: Sendable, Hashable {
    let datasetID: UUID
    let datasetGeneration: Int
    let modelGeneration: Int
    let runID: UUID
    let sessionID: UUID
    let targetID: UUID
    let expectedOrdinal: Int
    let historyRevision: Int
    let settingsRevision: Int

    /// True when both contexts describe the same dataset state, ignoring the target identity.
    func sharesState(with other: RoundContext) -> Bool {
        datasetID == other.datasetID
            && datasetGeneration == other.datasetGeneration
            && modelGeneration == other.modelGeneration
            && runID == other.runID
            && sessionID == other.sessionID
            && expectedOrdinal == other.expectedOrdinal
            && historyRevision == other.historyRevision
            && settingsRevision == other.settingsRevision
    }
}

/// An entry in progress, pinned to the target and forecast that were current when entry opened.
nonisolated struct EntryDraft: Sendable, Hashable, Identifiable {
    let submissionID: UUID
    let context: RoundContext
    let predictionID: UUID?

    var id: UUID { submissionID }
}

/// An immutable published forecast for one target.
nonisolated struct ForecastSnapshot: Sendable, Equatable, Identifiable {
    let id: UUID
    let context: RoundContext
    let values: [Int]
    /// One six-face distribution per die, in slot order.
    let probabilities: [[Double]]
    /// Smoothed per-position frequencies frozen before the target was observed.
    let baseline: [[Double]]
    let mode: PredictionMode
    let trainedPairs: Int
    let blendWeight: Double
    /// Walk-forward log likelihood ratio against fair dice when this forecast was published.
    let evidence: Double
    let publishedAt: Date

    var sum: Int { values.reduce(0, +) }

    /// True only when past out-of-sample performance is strong evidence against pure chance.
    var hasEvidence: Bool { mode != .random && evidence >= DiceRules.evidenceThreshold }
}

extension Array where Element == Double {
    nonisolated var encodedData: Data {
        withUnsafeBufferPointer { Data(buffer: $0) }
    }

    nonisolated init(encodedData data: Data) {
        let count = data.count / MemoryLayout<Double>.stride
        self.init(unsafeUninitializedCapacity: count) { buffer, initialized in
            _ = data.copyBytes(to: buffer)
            initialized = count
        }
    }
}

extension Array where Element == [Double] {
    nonisolated var flattened: [Double] { flatMap { $0 } }

    nonisolated init(unflattening values: [Double], rowLength: Int) {
        guard rowLength > 0 else {
            self = []
            return
        }
        self = stride(from: 0, to: values.count, by: rowLength).map {
            [Double](values[$0..<Swift.min($0 + rowLength, values.count)])
        }
    }
}
