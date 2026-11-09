import SwiftUI

/// One actual observed roll in the history list.
struct RollRow: View {
    let item: HistoryItem

    @ScaledMetric(relativeTo: .body) private var dieSize: CGFloat = 26
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    var body: some View {
        let stacked = dynamicTypeSize.isAccessibilitySize
        let layout = stacked
            ? AnyLayout(VStackLayout(alignment: .leading, spacing: 8))
            : AnyLayout(HStackLayout(alignment: .center, spacing: 12))

        layout {
            WrappingHStack(spacing: 6) {
                ForEach(Array(item.values.enumerated()), id: \.offset) { _, value in
                    DieFace(value: value, size: min(dieSize, 64))
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            VStack(alignment: stacked ? .leading : .trailing, spacing: 2) {
                HStack(spacing: 4) {
                    if item.isEdited {
                        Image(systemName: "pencil")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Text(item.sum, format: .number)
                        .font(.title3.weight(.semibold))
                        .monospacedDigit()
                }
                Text("#\(item.ordinal) · \(timestamp)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
        .alignmentGuide(.listRowSeparatorLeading) { _ in 0 }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(accessibilityText)
    }

    private var timestamp: String {
        Calendar.current.isDateInToday(item.observedAt)
            ? item.observedAt.formatted(date: .omitted, time: .shortened)
            : item.observedAt.formatted(.dateTime.month(.abbreviated).day().hour().minute())
    }

    private var accessibilityText: String {
        let text = String(localized: "Roll \(item.ordinal): \(item.values.spokenFaces). Sum \(item.sum). \(timestamp)")
        return item.isEdited ? text + ". " + String(localized: "Edited") : text
    }
}
