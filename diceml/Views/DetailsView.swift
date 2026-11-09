import SwiftUI

/// How the current prediction was made and how earlier predictions compared with chance.
struct DetailsView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    LabeledContent("Status", value: String(localized: statusText))
                    LabeledContent("Based on", value: String(localized: basisText))
                    LabeledContent("Rolls recorded", value: "\(model.observationCount)")
                    LabeledContent("Pattern influence", value: percent(model.card.forecast?.blendWeight ?? 0, digits: 0))
                    LabeledContent("Evidence vs. luck", value: evidenceText)
                    LabeledContent("Look back", value: String(localized: "\(model.dataset?.contextLength ?? 0) rolls"))
                }

                if let replay = model.replay {
                    ScoreSection(kind: "Replayed on past rolls", metrics: replay)
                }
                if let metrics = model.evaluation[.preEntry] {
                    ScoreSection(kind: "Shown before entry", metrics: metrics)
                }
                if let metrics = model.evaluation[.attested] {
                    ScoreSection(kind: "Shown before rolling", metrics: metrics)
                }
            }
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private var statusText: LocalizedStringResource {
        switch model.card {
        case .published(let forecast):
            if forecast.mode == .random { "Random" } else if forecast.hasEvidence { "Learned" } else { "Learning" }
        case .updating: "Updating"
        case .unavailable, .empty: "Unavailable"
        }
    }

    /// Names patterns only when they visibly contribute to the forecast.
    private var basisText: LocalizedStringResource {
        guard let forecast = model.card.forecast else { return "Unavailable" }
        switch forecast.mode {
        case .random: return "No rolls yet"
        case .sequence where forecast.blendWeight >= 0.005: return "Face counts and patterns"
        case .frequency, .sequence: return "Face counts"
        }
    }

    /// Walk-forward likelihood ratio against fair dice, shown against the threshold for "Learned".
    private var evidenceText: String {
        guard let forecast = model.card.forecast, forecast.mode != .random else { return "–" }
        let ratio = Foundation.exp(min(forecast.evidence, 700))
        let threshold = Foundation.exp(DiceRules.evidenceThreshold)
        let shown = ratio >= 1000
            ? ratio.formatted(.number.notation(.compactName).precision(.significantDigits(2)))
            : ratio.formatted(.number.precision(.fractionLength(1)))
        return String(localized: "\(shown)× of \(Int(threshold.rounded()))×")
    }
}

private struct ScoreSection: View {
    let kind: LocalizedStringResource
    let metrics: EvaluationMetrics

    var body: some View {
        Section {
            LabeledContent("Predictions", value: String(localized: kind))
            LabeledContent("Rolls checked", value: "\(metrics.count)")
            LabeledContent("Dice right", value: rateText(metrics.modelDieHitRate))
            LabeledContent("Most common face", value: rateText(metrics.baselineDieHitRate))
            LabeledContent("Pure chance", value: percent(1.0 / 6.0, digits: 1))
            LabeledContent("Probability vs. chance", value: probabilityText)
            LabeledContent("Whole roll right", value: String(localized: "\(metrics.modelExactRolls) of \(metrics.count)"))
            LabeledContent("Sum right", value: String(localized: "\(metrics.modelExactSums) of \(metrics.count)"))
        }
    }

    /// Average (geometric) probability given to the actual faces, relative to chance's 1 in 6.
    private var probabilityText: String {
        let ratio = Foundation.exp(EvaluationMetrics.uniformLogLoss - metrics.modelLogLoss) - 1
        return ratio.formatted(.percent.precision(.fractionLength(1)).sign(strategy: .always()))
    }

    private func rateText(_ rate: Double) -> String {
        "\(percent(rate, digits: 1)) ± \(percent(metrics.standardError(of: rate), digits: 1))"
    }
}

private func percent(_ value: Double, digits: Int) -> String {
    value.formatted(.percent.precision(.fractionLength(digits)))
}
