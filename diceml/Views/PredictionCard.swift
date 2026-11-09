import SwiftUI

/// The hero card: the next predicted faces and their sum, presented like a dark metal card.
struct PredictionCard: View {
    let state: AppModel.CardState
    let diceCount: Int

    @ScaledMetric(relativeTo: .largeTitle) private var dieSize: CGFloat = 58
    @ScaledMetric(relativeTo: .largeTitle) private var sumSize: CGFloat = 76
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var cardWidth: CGFloat = 0

    private var forecast: ForecastSnapshot? { state.forecast }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            StatusMark(state: state)

            Spacer(minLength: 24)

            WrappingHStack(spacing: 14) {
                ForEach(0..<diceCount, id: \.self) { index in
                    DieFace(
                        value: forecast?.values[safe: index],
                        size: min(dieSize, 112),
                        style: .ivory,
                        outline: Palette.champagne.opacity(0.35)
                    )
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .id(forecast?.id)
            .transition(reduceMotion ? .identity : .opacity.combined(with: .scale(scale: 0.96)))

            Spacer(minLength: 20)

            HStack(alignment: .lastTextBaseline) {
                Text("Sum")
                    .font(.footnote.weight(.semibold))
                    .textCase(.uppercase)
                    .tracking(1.6)
                    .foregroundStyle(Palette.champagne.opacity(0.7))
                Spacer()
                Text(forecast.map { "\($0.sum)" } ?? "–")
                    .font(.system(size: min(sumSize, 140), weight: .light, design: .serif))
                    .monospacedDigit()
                    .foregroundStyle(
                        LinearGradient(
                            colors: [Palette.champagne, Palette.champagneDeep],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .contentTransition(reduceMotion ? .identity : .numericText())
                    .lineLimit(1)
                    .minimumScaleFactor(0.6)
            }
        }
        .padding(26)
        .frame(maxWidth: .infinity, minHeight: cardMinimumHeight, alignment: .topLeading)
        .background(CardSurface())
        .clipShape(RoundedRectangle(cornerRadius: 30, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 30, style: .continuous)
                .strokeBorder(
                    LinearGradient(
                        colors: [.white.opacity(0.28), .white.opacity(0.04), Palette.champagne.opacity(0.22)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    lineWidth: 1
                )
        }
        .shadow(color: .black.opacity(0.28), radius: 22, y: 14)
        .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { cardWidth = $0 }
        .animation(reduceMotion ? nil : .smooth(duration: 0.45), value: forecast?.id)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(accessibilityText)
    }

    /// Credit-card proportions, capped so the card does not dominate wide iPad layouts.
    private var cardMinimumHeight: CGFloat {
        min(cardWidth / 1.586, 320)
    }

    private var accessibilityText: Text {
        switch state {
        case .published(let forecast):
            Text("Next roll prediction, \(String(localized: StatusMark.title(for: state))): \(forecast.values.spokenFaces). Sum \(forecast.sum)")
        case .updating:
            Text("Next roll prediction updating")
        case .unavailable:
            Text("Next roll prediction unavailable")
        case .empty:
            Text("Next roll prediction")
        }
    }
}

/// Small status indicator in the card's top corner.
private struct StatusMark: View {
    let state: AppModel.CardState

    static func title(for state: AppModel.CardState) -> LocalizedStringResource {
        switch state {
        case .published(let forecast):
            if forecast.mode == .random { "Random" } else if forecast.hasEvidence { "Learned" } else { "Learning" }
        case .updating: "Updating"
        case .unavailable: "Unavailable"
        case .empty: "Updating"
        }
    }

    var body: some View {
        HStack(spacing: 7) {
            indicator
            Text(Self.title(for: state))
                .font(.caption.weight(.semibold))
                .textCase(.uppercase)
                .tracking(1.4)
        }
        .foregroundStyle(.white.opacity(0.82))
        .frame(minHeight: 22)
        .opacity(state == .empty ? 0 : 1)
    }

    @ViewBuilder
    private var indicator: some View {
        switch state {
        case .published(let forecast) where forecast.hasEvidence:
            Circle().fill(Palette.champagne).frame(width: 7, height: 7)
        case .published(let forecast) where forecast.mode == .random:
            Circle().strokeBorder(.white.opacity(0.6), lineWidth: 1).frame(width: 7, height: 7)
        case .published:
            Circle().trim(from: 0, to: 0.5).stroke(Palette.champagne, lineWidth: 1.5).frame(width: 7, height: 7)
        case .updating:
            ProgressView().controlSize(.mini).tint(Palette.champagne)
        case .unavailable:
            Image(systemName: "exclamationmark").font(.caption2.weight(.bold))
        case .empty:
            EmptyView()
        }
    }
}

/// Brushed onyx surface with a soft specular sheen and a warm edge glow. Fully deterministic, so
/// re-rendering never changes its appearance.
private struct CardSurface: View {
    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Palette.onyxTop, Palette.onyxBottom],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            Canvas { context, size in
                var y: CGFloat = 0
                var line = 0
                while y < size.height {
                    let alpha = [0.035, 0.012, 0.022, 0.008][line % 4]
                    context.fill(Path(CGRect(x: 0, y: y, width: size.width, height: 0.6)), with: .color(.white.opacity(alpha)))
                    y += 1.6
                    line += 1
                }
            }
            LinearGradient(
                stops: [
                    .init(color: .white.opacity(0), location: 0.15),
                    .init(color: .white.opacity(0.09), location: 0.42),
                    .init(color: .white.opacity(0), location: 0.62),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            RadialGradient(
                colors: [Palette.champagne.opacity(0.16), .clear],
                center: .bottomTrailing,
                startRadius: 0,
                endRadius: 300
            )
        }
    }
}

extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

#Preview {
    VStack(spacing: 20) {
        PredictionCard(state: .updating, diceCount: 3)
        PredictionCard(state: .empty, diceCount: 5)
    }
    .padding()
}
