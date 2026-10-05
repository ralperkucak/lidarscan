import ARKit
import CoreImage
import simd
import SwiftUI

struct Keyframe: Codable {
    var id: Int
    var file: String
    var transform: [Float]          // ARKit kamera→dünya, 16 eleman (sütun-ana)
    var intrinsics: [Float]         // 9 eleman
    var center: SIMD3<Double> { SIMD3(Double(transform[12]), Double(transform[13]), Double(transform[14])) }
}

/// ARKit oturumu: LiDAR derinliğini dünya koordinatında biriktirir ve fotogrametri için anahtar kareler kaydeder.
final class CaptureManager: NSObject, ObservableObject, ARSessionDelegate {
    let session = ARSession()

    @Published var isScanning = false
    @Published var pointCount = 0
    @Published var keyframeCount = 0
    @Published var trackingText = "Hazır"

    // Ayarlar
    var voxelSize: Float = 0.01          // 1 cm
    var minDepth: Float = 0.25
    var maxDepth: Float = 4.5
    var maxKeyframes = 120
    var maxPoints = 8_000_000

    private struct Acc { var p = SIMD3<Float>.zero; var c = SIMD3<Float>.zero; var n: Float = 0 }
    private var voxels: [VoxelKey: Acc] = [:]
    private let work = DispatchQueue(label: "capture.work", qos: .userInitiated)
    private var busy = false
    private var lastDepthTime: TimeInterval = 0
    private var lastKeyPose: simd_float4x4?
    private(set) var keyframes: [Keyframe] = []
    private(set) var scanDir: URL?
    private let ciContext = CIContext()

    override init() { super.init(); session.delegate = self }

    static var isSupported: Bool {
        ARWorldTrackingConfiguration.isSupported &&
        ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    }

    // MARK: Kontrol

    func start() {
        let fm = FileManager.default
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let f = DateFormatter(); f.dateFormat = "yyyyMMdd_HHmmss"
        let dir = docs.appendingPathComponent("Scan_" + f.string(from: Date()))
        try? fm.createDirectory(at: dir.appendingPathComponent("images"), withIntermediateDirectories: true)
        scanDir = dir

        work.sync {
            voxels.removeAll(); keyframes.removeAll(); lastKeyPose = nil; lastDepthTime = 0; busy = false
        }
        pointCount = 0; keyframeCount = 0

        let cfg = ARWorldTrackingConfiguration()
        cfg.frameSemantics = [.sceneDepth]
        cfg.environmentTexturing = .none
        session.run(cfg, options: [.resetTracking, .removeExistingAnchors])
        isScanning = true
    }

    func stop() {
        session.pause()
        isScanning = false
        work.sync {}   // bekleyen işleri bitir
        if let dir = scanDir, let d = try? JSONEncoder().encode(keyframes) {
            try? d.write(to: dir.appendingPathComponent("keyframes.json"))
        }
    }

    /// Biriktirilmiş LiDAR bulutu
    func lidarCloud() -> PointCloud {
        work.sync {
            var c = PointCloud()
            c.positions.reserveCapacity(voxels.count); c.colors.reserveCapacity(voxels.count)
            for (_, a) in voxels where a.n > 0 {
                let p = a.p / a.n, col = a.c / a.n
                c.append(p, SIMD3<UInt8>(UInt8(min(255, max(0, col.x))),
                                         UInt8(min(255, max(0, col.y))),
                                         UInt8(min(255, max(0, col.z)))))
            }
            return c
        }
    }

    // MARK: ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        switch frame.camera.trackingState {
        case .normal: trackingText = isScanning ? "İzleme: iyi" : "Hazır"
        case .limited(let r): trackingText = "İzleme sınırlı: \(r)"
        case .notAvailable: trackingText = "İzleme yok"
        }
        guard isScanning, !busy, frame.camera.trackingState == .normal,
              frame.timestamp - lastDepthTime > 0.12, frame.sceneDepth != nil else { return }
        busy = true
        lastDepthTime = frame.timestamp
        work.async { [weak self] in
            self?.process(frame)
            self?.busy = false
        }
    }

    // MARK: İşleme

    private func process(_ frame: ARFrame) {
        guard let depth = frame.sceneDepth, let conf = depth.confidenceMap else { return }
        let dm = depth.depthMap
        let img = frame.capturedImage
        CVPixelBufferLockBaseAddress(dm, .readOnly)
        CVPixelBufferLockBaseAddress(conf, .readOnly)
        CVPixelBufferLockBaseAddress(img, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(dm, .readOnly)
            CVPixelBufferUnlockBaseAddress(conf, .readOnly)
            CVPixelBufferUnlockBaseAddress(img, .readOnly)
        }
        let dw = CVPixelBufferGetWidth(dm), dh = CVPixelBufferGetHeight(dm)
        let iw = CVPixelBufferGetWidth(img), ih = CVPixelBufferGetHeight(img)
        guard let dBase = CVPixelBufferGetBaseAddress(dm),
              let cBase = CVPixelBufferGetBaseAddress(conf),
              let yBase = CVPixelBufferGetBaseAddressOfPlane(img, 0),
              let uvBase = CVPixelBufferGetBaseAddressOfPlane(img, 1) else { return }
        let dStride = CVPixelBufferGetBytesPerRow(dm)
        let cStride = CVPixelBufferGetBytesPerRow(conf)
        let yStride = CVPixelBufferGetBytesPerRowOfPlane(img, 0)
        let uvStride = CVPixelBufferGetBytesPerRowOfPlane(img, 1)

        let K = frame.camera.intrinsics
        let fx = K[0][0], fy = K[1][1], cx = K[2][0], cy = K[2][1]
        let T = frame.camera.transform
        let sx = Float(iw) / Float(dw), sy = Float(ih) / Float(dh)

        for v in 0..<dh {
            let dRow = dBase.advanced(by: v * dStride).assumingMemoryBound(to: Float32.self)
            let cRow = cBase.advanced(by: v * cStride).assumingMemoryBound(to: UInt8.self)
            for u in 0..<dw {
                // Yalnızca yüksek güvenilirlikli LiDAR ölçümleri (ARConfidenceLevel.high = 2)
                if cRow[u] < UInt8(ARConfidenceLevel.high.rawValue) { continue }
                let d = dRow[u]
                if !(d > minDepth && d < maxDepth) { continue }
                let px = (Float(u) + 0.5) * sx, py = (Float(v) + 0.5) * sy
                // ARKit kamera uzayı: +x sağ, +y yukarı, -z ileri
                let pc = SIMD4<Float>((px - cx) / fx * d, -(py - cy) / fy * d, -d, 1)
                let w4 = T * pc
                let pw = SIMD3<Float>(w4.x, w4.y, w4.z)

                let key = VoxelKey(pw, size: voxelSize)
                var a = voxels[key] ?? Acc()
                if a.n >= 24 { continue }                    // doymuş voksel
                if a.n == 0 && voxels.count >= maxPoints { continue }

                let ix = min(iw - 1, Int(px)), iy = min(ih - 1, Int(py))
                let Y = Float(yBase.load(fromByteOffset: iy * yStride + ix, as: UInt8.self))
                let uvOff = (iy / 2) * uvStride + (ix / 2) * 2
                let Cb = Float(uvBase.load(fromByteOffset: uvOff, as: UInt8.self)) - 128
                let Cr = Float(uvBase.load(fromByteOffset: uvOff + 1, as: UInt8.self)) - 128
                let col = SIMD3<Float>(Y + 1.402 * Cr, Y - 0.344136 * Cb - 0.714136 * Cr, Y + 1.772 * Cb)

                a.p += pw; a.c += col; a.n += 1
                voxels[key] = a
            }
        }
        let count = voxels.count
        DispatchQueue.main.async { self.pointCount = count }

        maybeSaveKeyframe(frame)
    }

    private func maybeSaveKeyframe(_ frame: ARFrame) {
        guard keyframes.count < maxKeyframes, let dir = scanDir else { return }
        let T = frame.camera.transform
        if let last = lastKeyPose {
            let dist = simd_distance(T.columns.3, last.columns.3)
            // bakış yönleri arası açı
            let f0 = -SIMD3<Float>(last.columns.2.x, last.columns.2.y, last.columns.2.z)
            let f1 = -SIMD3<Float>(T.columns.2.x, T.columns.2.y, T.columns.2.z)
            let ang = acos(max(-1, min(1, simd_dot(f0, f1)))) * 180 / .pi
            if dist < 0.10 && ang < 12 { return }
        }
        let ci = CIImage(cvPixelBuffer: frame.capturedImage)
        guard let jpg = ciContext.jpegRepresentation(of: ci, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                                                     options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.92])
        else { return }
        let id = keyframes.count
        let name = String(format: "img_%04d.jpg", id)
        do { try jpg.write(to: dir.appendingPathComponent("images").appendingPathComponent(name)) } catch { return }
        let K = frame.camera.intrinsics
        keyframes.append(Keyframe(id: id, file: name,
                                  transform: (0..<4).flatMap { c in (0..<4).map { r in T[c][r] } },
                                  intrinsics: (0..<3).flatMap { c in (0..<3).map { r in K[c][r] } }))
        lastKeyPose = T
        let n = keyframes.count
        DispatchQueue.main.async { self.keyframeCount = n }
    }
}
