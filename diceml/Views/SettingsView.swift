import SwiftUI

/// Table, history, and learning controls, grouped by layout rather than section titles.
struct SettingsView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var showingNewTable = false
    @State private var confirmingErase = false
    @State private var confirmingDelete = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("Table", selection: tableSelection) {
                        ForEach(model.datasets) { table in
                            Text("\(table.name) · \(table.diceCount) dice").tag(Optional(table.id))
                        }
                    }
                    Button("New table") { showingNewTable = true }
                }

                Section {
                    Stepper(value: lookBack, in: DiceRules.contextLengthRange) {
                        LabeledContent("Look back", value: String(localized: "\(model.dataset?.contextLength ?? 0) rolls"))
                    }
                    Button("Start new session") {
                        model.startSession()
                        dismiss()
                    }
                }

                Section {
                    Button {
                        model.trainNewModel()
                    } label: {
                        HStack {
                            Text("Restart learning")
                            Spacer()
                            if model.isTraining {
                                ProgressView()
                            }
                        }
                    }
                    Button("Erase all rolls", role: .destructive) {
                        confirmingErase = true
                    }
                    Button("Delete table", role: .destructive) {
                        confirmingDelete = true
                    }
                } footer: {
                    VStack(spacing: 6) {
                        Text(Self.versionText)
                        Text("For entertainment only. Dice outcomes are random and predictions are never guaranteed. You are solely responsible for any decisions or bets you make.")
                    }
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity)
                    .padding(.top, 24)
                }
            }
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .confirmationDialog(
                "Erase all rolls in “\(model.dataset?.name ?? "")”?",
                isPresented: $confirmingErase,
                titleVisibility: .visible
            ) {
                Button("Erase \(model.observationCount) rolls", role: .destructive) {
                    model.clearObservations()
                    dismiss()
                }
            } message: {
                Text("Removes this table’s rolls, predictions, and learned patterns. Other tables are kept.")
            }
            .confirmationDialog(
                "Delete “\(model.dataset?.name ?? "")”?",
                isPresented: $confirmingDelete,
                titleVisibility: .visible
            ) {
                Button("Delete table", role: .destructive) {
                    model.deleteCurrentTable()
                    dismiss()
                }
            } message: {
                Text("Removes this table and all of its rolls. This can’t be undone.")
            }
            .sheet(isPresented: $showingNewTable) {
                NewTableView()
            }
        }
    }

    /// Marketing version and build number, such as "1.0 (1)".
    private static var versionText: String {
        let info = Bundle.main.infoDictionary
        let version = info?["CFBundleShortVersionString"] as? String ?? "–"
        let build = info?["CFBundleVersion"] as? String ?? "–"
        return "\(version) (\(build))"
    }

    private var tableSelection: Binding<UUID?> {
        Binding(
            get: { model.dataset?.id },
            set: { if let id = $0 { model.selectDataset(id) } }
        )
    }

    private var lookBack: Binding<Int> {
        Binding(
            get: { model.dataset?.contextLength ?? DiceRules.defaultContextLength },
            set: { model.setContextLength($0) }
        )
    }
}

private struct NewTableView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var name = ""
    @State private var diceCount = DiceRules.defaultDiceCount

    var body: some View {
        NavigationStack {
            Form {
                TextField("Name", text: $name, prompt: Text(model.suggestedTableName))
                Stepper(value: $diceCount, in: DiceRules.diceCountRange) {
                    LabeledContent("Dice", value: "\(diceCount)")
                }
            }
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Create") {
                        model.createDataset(name: name, diceCount: diceCount)
                        dismiss()
                    }
                }
            }
        }
        .presentationDetents([.medium])
    }
}
