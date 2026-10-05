import SwiftUI
import ARKit
import SceneKit
import Combine

@main
struct LiDARScanApp: App {
    var body: some Scene { WindowGroup { ContentView() } }
}

struct ARPreview: UIViewRepresentable {
    let session: ARSession
    func makeUIView(context: Context) -> ARSCNView {
        let v = ARSCNView(frame: .zero)
        v.session = session
        v.automaticallyUpdatesLighting = false
        return v
    }
    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

// MARK: - Katman modeli

/// Her veri kaynağı kendi katmanında bellekte tutulur; birbirini değiştirmez.
struct Layer: Identifiable {
    enum Kind { case lidar, photoRaw, photoAligned }
    let id = UUID()
    var kind: Kind
    var name: String
    var cloud: PointCloud
    var selected = true
}

enum ExportFormat: String, CaseIterable, Identifiable {
    case ply = "PLY", las = "LAS"
    var id: String { rawValue }
    var ext: String { rawValue.lowercased() }
}

@MainActor
final class AppModel: ObservableObject {
    enum Stage { case idle, scanning, ready }
    @Published var stage: Stage = .idle
    @Published var layers: [Layer] = []
    @Published var busy = false
    @Published var progress = 0.0
    @Published var message = ""
    @Published var report = ""
    @Published var exportURLs: [URL] = []

    // Dışa aktarma seçenekleri
    @Published var format: ExportFormat = .las
    @Published var mergeSelected = false

    let capture = CaptureManager()
    private var photoRecon: PhotoReconstruction?
    private var bag = Set<AnyCancellable>()

    init() {
        // CaptureManager'daki nokta/kare sayacı değişince arayüz yenilensin
        capture.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &bag)
    }

    // MARK: Tarama

    func startScan() {
        layers.removeAll(); photoRecon = nil; exportURLs = []; report = ""
        capture.start(); stage = .scanning
        message = "Ortamı yavaşça, örtüşmeli ve her açıdan dolaşın."
    }

    func stopScan() {
        capture.stop()
        let c = capture.lidarCloud()
        layers = [Layer(kind: .lidar, name: "LiDAR", cloud: c)]
        stage = .ready
        message = "LiDAR katmanı hazır (\(c.count.formatted()) nokta)."
    }

    // MARK: Fotogrametri (ayrı katman)

    func runPhotogrammetry() {
        guard let dir = capture.scanDir else { return }
        let kfs = capture.keyframes
        busy = true; progress = 0
        Task {
            defer { busy = false }
            do {
                guard #available(iOS 17.0, *) else { throw ProcessingError.unsupported }
                let rec = try await PhotogrammetryProcessor.reconstruct(scanDir: dir, keyframes: kfs) { f, m in
                    Task { @MainActor in self.progress = f; self.message = m }
                }
                photoRecon = rec
                layers.removeAll { $0.kind == .photoRaw || $0.kind == .photoAligned }
                layers.append(Layer(kind: .photoRaw, name: "Fotogrametri (ham, ölçeksiz)", cloud: rec.cloud, selected: false))
                message = "Fotogrametri katmanı eklendi (\(rec.cloud.count.formatted()) nokta). Şimdi ICP ile hizalayın."
            } catch { message = "Hata: \(error.localizedDescription)" }
        }
    }

    // MARK: ICP hizalama (yeni katman üretir, kaynakları korur)

    func alignWithICP() {
        guard let rec = photoRecon, let lidar = layers.first(where: { $0.kind == .lidar })?.cloud else { return }
        let kfs = capture.keyframes
        busy = true; message = "Hizalanıyor (poz + ICP)…"
        Task {
            defer { busy = false }
            do {
                guard #available(iOS 17.0, *) else { throw ProcessingError.unsupported }
                let out = try await Task.detached(priority: .userInitiated) {
                    try PhotogrammetryProcessor.align(photo: rec, keyframes: kfs, lidar: lidar)
                }.value
                layers.removeAll { $0.kind == .photoAligned }
                layers.append(Layer(kind: .photoAligned, name: "Fotogrametri (LiDAR'a hizalı)", cloud: out.aligned))
                if let i = layers.firstIndex(where: { $0.kind == .photoRaw }) { layers[i].selected = false }
                report = out.report; message = "Hizalama tamamlandı."
            } catch { message = "Hata: \(error.localizedDescription)" }
        }
    }

    // MARK: Dışa aktarma — kullanıcı katmanı ve biçimi seçer

    func export() {
        guard let base = capture.scanDir else { return }
        let sel = layers.filter { $0.selected && $0.cloud.count > 0 }
        guard !sel.isEmpty else { message = "Dışa aktarmak için en az bir katman seçin."; return }
        let dir = base.appendingPathComponent("exports")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let fmt = format, merge = mergeSelected
        busy = true; message = "Dosyalar yazılıyor…"
        Task {
            defer { busy = false }
            do {
                let urls = try await Task.detached(priority: .userInitiated) { () -> [URL] in
                    var jobs: [(String, PointCloud)] = []
                    if merge {
                        var all = PointCloud()
                        for l in sel { all.append(contentsOf: l.cloud) }
                        jobs = [("merged", all)]
                    } else {
                        jobs = sel.map { (Self.fileName($0), $0.cloud) }
                    }
                    var out: [URL] = []
                    for (name, cloud) in jobs {
                        let url = dir.appendingPathComponent(name + "." + fmt.ext)
                        if fmt == .ply { try PLYWriter.write(cloud, to: url) } else { try LASWriter.write(cloud, to: url) }
                        out.append(url)
                    }
                    return out
                }.value
                exportURLs = urls
                message = "\(urls.count) dosya hazır."
            } catch { message = "Hata: \(error.localizedDescription)" }
        }
    }

    nonisolated private static func fileName(_ l: Layer) -> String {
        switch l.kind {
        case .lidar: return "lidar"
        case .photoRaw: return "photo_raw"
        case .photoAligned: return "photo_aligned"
        }
    }
}

// MARK: - Arayüz

struct ContentView: View {
    @StateObject private var m = AppModel()
    @State private var showExport = false

    var body: some View {
        ZStack(alignment: .bottom) {
            if CaptureManager.isSupported {
                ARPreview(session: m.capture.session).ignoresSafeArea()
            } else {
                Color.black.ignoresSafeArea()
                Text("Bu cihazda LiDAR yok.").foregroundColor(.white)
            }
            panel
        }
        .sheet(isPresented: $showExport) { ExportSheet(m: m) }
    }

    var panel: some View {
        VStack(spacing: 10) {
            if m.stage == .scanning {
                HStack {
                    Label("\(m.capture.pointCount.formatted()) nokta", systemImage: "circle.grid.3x3")
                    Spacer()
                    Label("\(m.capture.keyframeCount) kare", systemImage: "photo.on.rectangle")
                }.font(.footnote.monospacedDigit())
                Text(m.capture.trackingText).font(.caption)
            }
            if !m.message.isEmpty { Text(m.message).font(.footnote).multilineTextAlignment(.center) }
            if m.busy { ProgressView(value: m.progress > 0 ? m.progress : nil) }
            if !m.report.isEmpty {
                ScrollView { Text(m.report).font(.system(size: 10, design: .monospaced)).frame(maxWidth: .infinity, alignment: .leading) }
                    .frame(maxHeight: 110)
            }
            if m.stage == .ready { layerList }
            controls
        }
        .padding().background(.ultraThinMaterial).cornerRadius(18).padding()
    }

    var layerList: some View {
        VStack(spacing: 4) {
            ForEach($m.layers) { $l in
                Toggle(isOn: $l.selected) {
                    HStack { Text(l.name).font(.footnote); Spacer(); Text(l.cloud.count.formatted()).font(.caption.monospacedDigit()) }
                }.toggleStyle(.switch).controlSize(.mini)
            }
        }
    }

    @ViewBuilder var controls: some View {
        switch m.stage {
        case .idle:
            Button("Taramayı Başlat") { m.startScan() }.buttonStyle(.borderedProminent)
        case .scanning:
            Button("Taramayı Bitir") { m.stopScan() }.buttonStyle(.borderedProminent).tint(.red)
        case .ready:
            HStack {
                Button("1. Fotoğraftan Üret") { m.runPhotogrammetry() }
                    .disabled(m.busy)
                Button("2. ICP Hizala") { m.alignWithICP() }
                    .disabled(m.busy || !m.layers.contains { $0.kind == .photoRaw })
            }.buttonStyle(.bordered)
            HStack {
                Button("Dışa Aktar…") { showExport = true }.buttonStyle(.borderedProminent).disabled(m.busy)
                Button("Yeni Tarama") { m.startScan() }.buttonStyle(.bordered).disabled(m.busy)
            }
        }
    }
}

struct ExportSheet: View {
    @ObservedObject var m: AppModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Katmanlar") {
                    ForEach($m.layers) { $l in
                        Toggle(isOn: $l.selected) {
                            VStack(alignment: .leading) {
                                Text(l.name)
                                Text("\(l.cloud.count.formatted()) nokta").font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                Section("Biçim") {
                    Picker("Dosya türü", selection: $m.format) {
                        ForEach(ExportFormat.allCases) { Text($0.rawValue).tag($0) }
                    }.pickerStyle(.segmented)
                    Toggle("Seçili katmanları tek dosyada birleştir", isOn: $m.mergeSelected)
                }
                Section {
                    Button("Dosyaları Oluştur") { m.export() }.disabled(m.busy)
                    if m.busy { ProgressView() }
                    Text(m.message).font(.footnote)
                    if !m.exportURLs.isEmpty {
                        ShareLink(items: m.exportURLs) { Label("Paylaş / Dosyalar'a Kaydet", systemImage: "square.and.arrow.up") }
                    }
                }
            }
            .navigationTitle("Dışa Aktar")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Bitti") { dismiss() } } }
        }
    }
}
