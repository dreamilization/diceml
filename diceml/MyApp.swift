import SwiftUI

@main struct MyApp: App {
    @State private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(model)
                .task {
                    if model.loadState == .loading {
                        model.load()
                    }
                }
        }
    }
}
