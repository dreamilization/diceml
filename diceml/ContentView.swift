import SwiftUI

struct ContentView: View {
    @Environment(AppModel.self) private var model
    @State private var entry: EntryPresentation?
    @State private var showingSettings = false
    @State private var showingDetails = false

    private let contentWidth: CGFloat = 760

    var body: some View {
        NavigationStack {
            Group {
                switch model.loadState {
                case .loading:
                    ProgressView()
                case .failed(let message):
                    ContentUnavailableView {
                        Label("History unavailable", systemImage: "exclamationmark.triangle")
                    } description: {
                        Text(message)
                    } actions: {
                        Button("Retry") { model.load() }
                            .buttonStyle(.borderedProminent)
                    }
                case .ready:
                    main
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(.systemGroupedBackground))
            .navigationTitle("DiceML")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                if model.loadState == .ready {
                    ToolbarItemGroup(placement: .topBarTrailing) {
                        Button("Details", systemImage: "chart.bar.xaxis") { showingDetails = true }
                        Button("Settings", systemImage: "gearshape") { showingSettings = true }
                    }
                }
            }
        }
        .sheet(item: $entry, onDismiss: model.cancelEntry) { presentation in
            entrySheet(presentation)
        }
        .sheet(isPresented: $showingSettings) { SettingsView() }
        .sheet(isPresented: $showingDetails) { DetailsView() }
        .alert("Something went wrong", isPresented: errorBinding) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.errorMessage ?? "")
        }
    }

    private var main: some View {
        List {
            Section {
                PredictionCard(state: model.card, diceCount: model.dataset?.diceCount ?? DiceRules.defaultDiceCount)
                    .listRowInsets(EdgeInsets())
                    .listRowBackground(Color.clear)
            }

            ForEach(model.historySections) { section in
                Section {
                    ForEach(section.items) { item in
                        RollRow(item: item)
                            .swipeActions {
                                Button("Delete", systemImage: "trash", role: .destructive) {
                                    model.deleteRoll(id: item.id)
                                }
                                Button("Edit", systemImage: "pencil") {
                                    entry = .edit(item)
                                }
                            }
                            .contextMenu {
                                Button("Edit", systemImage: "pencil") { entry = .edit(item) }
                                Button("Delete", systemImage: "trash", role: .destructive) {
                                    model.deleteRoll(id: item.id)
                                }
                            }
                    }
                }
            }
        }
        .listStyle(.insetGrouped)
        .scrollContentBackground(.hidden)
        .frame(maxWidth: contentWidth)
        .safeAreaInset(edge: .bottom) {
            Button {
                if let draft = model.beginEntry() {
                    entry = .new(draft)
                }
            } label: {
                Label("Add roll", systemImage: "plus")
                    .font(.headline)
                    .frame(maxWidth: .infinity, minHeight: 52)
            }
            .buttonStyle(PrimaryCapsuleButtonStyle())
            .disabled(model.round == nil)
            .padding(.horizontal, 20)
            .padding(.bottom, 8)
            .frame(maxWidth: contentWidth)
        }
    }

    @ViewBuilder
    private func entrySheet(_ presentation: EntryPresentation) -> some View {
        let diceCount = model.dataset?.diceCount ?? DiceRules.defaultDiceCount
        switch presentation {
        case .new(let draft):
            // Attesting timing only matters for learned forecasts; random cold-start forecasts are never scored.
            let canAttest = draft.predictionID != nil && model.card.forecast?.mode != .random
            EntrySheet(diceCount: diceCount, canAttest: canAttest) { values, attested in
                try model.saveEntry(values: values, attested: attested)
            }
        case .edit(let item):
            EntrySheet(diceCount: item.values.count, initialValues: item.values) { values, _ in
                try model.editRoll(id: item.id, values: values)
            }
        }
    }

    private var errorBinding: Binding<Bool> {
        Binding(
            get: { model.errorMessage != nil },
            set: { if !$0 { model.errorMessage = nil } }
        )
    }
}

/// High-contrast monochrome capsule that pairs with the dark prediction card in either appearance.
private struct PrimaryCapsuleButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    @Environment(\.colorScheme) private var colorScheme

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(colorScheme == .dark ? Palette.onyxBottom : Palette.ivoryLight)
            .background(
                Capsule().fill(colorScheme == .dark ? Palette.ivory : Palette.onyxTop)
            )
            .opacity(isEnabled ? (configuration.isPressed ? 0.75 : 1) : 0.4)
            .contentShape(Capsule())
    }
}

private enum EntryPresentation: Identifiable {
    case new(EntryDraft)
    case edit(HistoryItem)

    var id: UUID {
        switch self {
        case .new(let draft): draft.submissionID
        case .edit(let item): item.id
        }
    }
}

#Preview {
    let model = AppModel(storeFactory: { try DiceStore.inMemory() })
    ContentView()
        .environment(model)
        .task { model.load() }
}
