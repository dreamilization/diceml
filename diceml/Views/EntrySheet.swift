import SwiftUI

/// Ordered slot entry for one actual roll. Values are never prefilled for a new roll.
struct EntrySheet: View {
    let diceCount: Int
    /// Shows the attestation toggle when a published forecast was pinned to this entry.
    let canAttest: Bool
    let onSave: (_ values: [Int], _ attested: Bool) throws -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var slots: [Int?]
    @State private var activeSlot = 0
    @State private var attested = false
    @State private var saveError: String?

    @ScaledMetric(relativeTo: .title) private var slotSize: CGFloat = 52
    @ScaledMetric(relativeTo: .title) private var pickerSize: CGFloat = 48
    @ScaledMetric(relativeTo: .largeTitle) private var sumSize: CGFloat = 44

    init(
        diceCount: Int,
        initialValues: [Int]? = nil,
        canAttest: Bool = false,
        onSave: @escaping (_ values: [Int], _ attested: Bool) throws -> Void
    ) {
        self.diceCount = diceCount
        self.canAttest = canAttest
        self.onSave = onSave
        let initial = initialValues.map { $0.map(Optional.some) } ?? Array(repeating: nil, count: diceCount)
        _slots = State(initialValue: initial)
    }

    private var values: [Int]? {
        let filled = slots.compactMap { $0 }
        return DiceRules.isValid(filled, diceCount: diceCount) ? filled : nil
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 28) {
                    WrappingHStack(spacing: 10) {
                        ForEach(0..<diceCount, id: \.self) { index in
                            slotButton(index)
                        }
                    }

                    Text(values.map { "\($0.reduce(0, +))" } ?? "–")
                        .font(.system(size: min(sumSize, 96), weight: .bold, design: .rounded))
                        .monospacedDigit()
                        .accessibilityLabel(values.map { Text("Sum \($0.reduce(0, +))") } ?? Text("Sum incomplete"))

                    WrappingHStack(spacing: 12) {
                        ForEach(DiceRules.faces, id: \.self) { face in
                            Button {
                                select(face)
                            } label: {
                                DieFace(value: face, size: max(pickerSize, 44))
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel("\(face)")
                        }
                    }

                    if canAttest {
                        Toggle("Saw prediction before rolling", isOn: $attested)
                    }

                    if let saveError {
                        Text(saveError)
                            .font(.callout)
                            .foregroundStyle(.red)
                            .multilineTextAlignment(.center)
                    }
                }
                .padding(24)
                .frame(maxWidth: 560)
                .frame(maxWidth: .infinity)
            }
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save", action: save)
                        .disabled(values == nil)
                }
            }
        }
        .presentationDetents([.medium, .large])
    }

    private func slotButton(_ index: Int) -> some View {
        Button {
            activeSlot = index
        } label: {
            DieFace(value: slots[index], size: max(slotSize, 44))
                .padding(4)
                .overlay {
                    RoundedRectangle(cornerRadius: max(slotSize, 44) * 0.3, style: .continuous)
                        .stroke(Color.accentColor, lineWidth: index == activeSlot ? 3 : 0)
                }
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Die \(index + 1)")
        .accessibilityValue(slots[index].map { Text(verbatim: "\($0)") } ?? Text("Empty"))
        .accessibilityAddTraits(index == activeSlot ? .isSelected : [])
    }

    /// Fills the active slot and moves to the next unfilled slot, if any.
    private func select(_ face: Int) {
        slots[activeSlot] = face
        saveError = nil
        let order = Array(activeSlot + 1..<diceCount) + Array(0..<activeSlot)
        if let next = order.first(where: { slots[$0] == nil }) {
            activeSlot = next
        }
    }

    private func save() {
        guard let values else { return }
        do {
            try onSave(values, canAttest && attested)
            dismiss()
        } catch {
            saveError = error.localizedDescription
        }
    }
}

#Preview {
    EntrySheet(diceCount: 3, canAttest: true) { _, _ in }
}
