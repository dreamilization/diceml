import SwiftUI

/// A vector die face. `nil` draws an unfilled slot outline.
struct DieFace: View {
    enum Style {
        /// Flat face for lists and controls, adapting to light and dark appearance.
        case flat
        /// Polished ivory die for the prediction card.
        case ivory
    }

    let value: Int?
    var size: CGFloat = 44
    var style: Style = .flat
    var outline: Color = .secondary

    var body: some View {
        Canvas { context, canvasSize in
            let rect = CGRect(origin: .zero, size: canvasSize)
            let shape = RoundedRectangle(cornerRadius: canvasSize.width * 0.22, style: .continuous)
            guard let value else {
                context.stroke(
                    shape.path(in: rect.insetBy(dx: 1, dy: 1)),
                    with: .color(outline),
                    style: StrokeStyle(lineWidth: 1.5, dash: [5, 4])
                )
                return
            }
            switch style {
            case .flat:
                context.fill(shape.path(in: rect), with: .color(Color(.systemGray5)))
            case .ivory:
                context.fill(shape.path(in: rect), with: .linearGradient(
                    Gradient(colors: [Palette.ivoryLight, Palette.ivory, Palette.ivoryShade]),
                    startPoint: .zero,
                    endPoint: CGPoint(x: canvasSize.width, y: canvasSize.height)
                ))
                context.stroke(
                    shape.path(in: rect.insetBy(dx: 0.75, dy: 0.75)),
                    with: .linearGradient(
                        Gradient(colors: [.white.opacity(0.9), .white.opacity(0)]),
                        startPoint: .zero,
                        endPoint: CGPoint(x: 0, y: canvasSize.height * 0.6)
                    ),
                    lineWidth: 1.5
                )
            }

            let diameter = canvasSize.width * 0.18
            for point in Self.pips(for: value) {
                let pipRect = CGRect(
                    x: point.x * canvasSize.width - diameter / 2,
                    y: point.y * canvasSize.height - diameter / 2,
                    width: diameter,
                    height: diameter
                )
                switch style {
                case .flat:
                    context.fill(Circle().path(in: pipRect), with: .color(.primary))
                case .ivory:
                    // Recessed pips: darker at the upper edge, as if drilled into the face.
                    context.fill(Circle().path(in: pipRect), with: .linearGradient(
                        Gradient(colors: [Palette.pipShadow, Palette.pip]),
                        startPoint: CGPoint(x: pipRect.midX, y: pipRect.minY),
                        endPoint: CGPoint(x: pipRect.midX, y: pipRect.maxY)
                    ))
                }
            }
        }
        .frame(width: size, height: size)
        .shadow(color: style == .ivory && value != nil ? .black.opacity(0.45) : .clear, radius: size * 0.08, y: size * 0.06)
        .accessibilityHidden(true)
    }

    /// Pip centers in unit coordinates.
    static func pips(for value: Int) -> [CGPoint] {
        let low = 0.27, mid = 0.5, high = 0.73
        let center = CGPoint(x: mid, y: mid)
        let corners = [CGPoint(x: low, y: low), CGPoint(x: high, y: high)]
        let otherCorners = [CGPoint(x: high, y: low), CGPoint(x: low, y: high)]
        switch value {
        case 1: return [center]
        case 2: return corners
        case 3: return corners + [center]
        case 4: return corners + otherCorners
        case 5: return corners + otherCorners + [center]
        case 6: return corners + otherCorners + [CGPoint(x: low, y: mid), CGPoint(x: high, y: mid)]
        default: return []
        }
    }
}

/// Shared colors for the prediction card and its dice.
enum Palette {
    static let onyxTop = Color(red: 0.165, green: 0.161, blue: 0.157)
    static let onyxBottom = Color(red: 0.055, green: 0.055, blue: 0.059)
    static let champagne = Color(red: 0.902, green: 0.816, blue: 0.616)
    static let champagneDeep = Color(red: 0.749, green: 0.627, blue: 0.396)
    static let ivoryLight = Color(red: 0.996, green: 0.988, blue: 0.965)
    static let ivory = Color(red: 0.957, green: 0.937, blue: 0.894)
    static let ivoryShade = Color(red: 0.871, green: 0.839, blue: 0.776)
    static let pip = Color(red: 0.11, green: 0.10, blue: 0.10)
    static let pipShadow = Color(red: 0.02, green: 0.02, blue: 0.02)
}

/// Places subviews left to right and wraps top to bottom, preserving reading order.
struct WrappingHStack: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = arrange(subviews, width: proposal.width ?? .infinity)
        let width = rows.map(\.width).max() ?? 0
        let height = rows.map(\.height).reduce(0, +) + spacing * CGFloat(max(rows.count - 1, 0))
        return CGSize(width: width, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var y = bounds.minY
        for row in arrange(subviews, width: bounds.width) {
            var x = bounds.minX
            for index in row.indices {
                let size = subviews[index].sizeThatFits(.unspecified)
                subviews[index].place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
            y += row.height + spacing
        }
    }

    private struct Row {
        var indices: [Int] = []
        var width: CGFloat = 0
        var height: CGFloat = 0
    }

    private func arrange(_ subviews: Subviews, width: CGFloat) -> [Row] {
        var rows: [Row] = []
        var current = Row()
        for index in subviews.indices {
            let size = subviews[index].sizeThatFits(.unspecified)
            let proposed = current.indices.isEmpty ? size.width : current.width + spacing + size.width
            if !current.indices.isEmpty, proposed > width {
                rows.append(current)
                current = Row()
            }
            current.width = current.indices.isEmpty ? size.width : current.width + spacing + size.width
            current.height = max(current.height, size.height)
            current.indices.append(index)
        }
        if !current.indices.isEmpty { rows.append(current) }
        return rows
    }
}

extension Array where Element == Int {
    /// Spoken form such as "3, 5, 2".
    var spokenFaces: String {
        map(String.init).joined(separator: ", ")
    }
}

#Preview {
    VStack(spacing: 20) {
        WrappingHStack {
            ForEach(1...6, id: \.self) { DieFace(value: $0) }
            DieFace(value: nil)
        }
        WrappingHStack(spacing: 12) {
            ForEach(1...6, id: \.self) { DieFace(value: $0, size: 56, style: .ivory) }
        }
        .padding()
        .background(Palette.onyxBottom)
    }
    .padding()
}
