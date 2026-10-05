import Foundation
import RealityKit
import CoreImage
import CoreVideo
import simd

enum ProcessingError: LocalizedError {
    case unsupported, tooFewImages, noPointCloud, alignmentFailed, noLayer
    var errorDescription: String? {
        switch self {
        case .unsupported: return "Bu cihaz Object Capture (PhotogrammetrySession) desteklemiyor."
        case .tooFewImages: return "Fotogrametri için en az 10 anahtar kare gerekir."
        case .noPointCloud: return "Fotogrametri nokta bulutu üretemedi."
        case .alignmentFailed: return "Kamera pozlarından ilk hizalama yapılamadı (yetersiz ortak poz)."
        case .noLayer: return "Hizalama için hem LiDAR hem fotogrametri katmanı gerekir."
        }
    }
}

/// Fotogrametri sonucu: ham (ölçeksiz, kendi koordinatında) bulut + kamera pozları.
struct PhotoReconstruction {
    var cloud: PointCloud
    var poses: [Int: simd_float4x4]
}

struct AlignmentOutcome {
    var aligned: PointCloud
    var report: String
}

@available(iOS 17.0, *)
enum PhotogrammetryProcessor {

    static var isSupported: Bool { PhotogrammetrySession.isSupported }

    // MARK: 1) Fotoğraflardan nokta bulutu (ham)

    static func reconstruct(scanDir: URL, keyframes: [Keyframe],
                            progress: @escaping @Sendable (Double, String) -> Void) async throws -> PhotoReconstruction {
        guard isSupported else { throw ProcessingError.unsupported }
        guard keyframes.count >= 10 else { throw ProcessingError.tooFewImages }

        let imagesDir = scanDir.appendingPathComponent("images")
        let ctx = CIContext()
        let samples = keyframes.lazy.compactMap { kf -> PhotogrammetrySample? in
            guard let ci = CIImage(contentsOf: imagesDir.appendingPathComponent(kf.file)) else { return nil }
            var pb: CVPixelBuffer?
            CVPixelBufferCreate(nil, Int(ci.extent.width), Int(ci.extent.height), kCVPixelFormatType_32BGRA,
                                [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &pb)
            guard let buf = pb else { return nil }
            ctx.render(ci, to: buf)
            return PhotogrammetrySample(id: kf.id, image: buf)
        }

        var cfg = PhotogrammetrySession.Configuration()
        cfg.featureSensitivity = .high
        cfg.sampleOrdering = .sequential
        cfg.isObjectMaskingEnabled = false

        let session = try PhotogrammetrySession(input: samples, configuration: cfg)
        try session.process(requests: [.poses, .pointCloud])

        var poses: [Int: simd_float4x4] = [:]
        var cloud = PointCloud()
        progress(0, "Fotogrametri başladı…")
        outer: for try await out in session.outputs {
            switch out {
            case .requestProgress(_, let f):
                progress(f, String(format: "Fotogrametri %%%.0f", f * 100))
            case .requestComplete(_, let result):
                switch result {
                case .poses(let p): poses = p.posesBySample.mapValues { $0.transform }
                case .pointCloud(let pc):
                    cloud.positions.reserveCapacity(pc.points.count)
                    for pt in pc.points {
                        let c = pt.color                      // SIMD4<Float> 0…1 varsayımı
                        cloud.append(pt.position, SIMD3<UInt8>(UInt8(max(0, min(255, c.x * 255))),
                                                               UInt8(max(0, min(255, c.y * 255))),
                                                               UInt8(max(0, min(255, c.z * 255)))))
                    }
                default: break
                }
            case .requestError(_, let e): throw e
            case .processingComplete: break outer
            default: break
            }
        }
        guard cloud.count > 0 else { throw ProcessingError.noPointCloud }
        return PhotoReconstruction(cloud: cloud, poses: poses)
    }

    // MARK: 2) Hizalama: kamera pozları (ölçek) + ICP

    /// Katmanları değiştirmez; hizalanmış yeni bir fotogrametri bulutu döndürür.
    static func align(photo: PhotoReconstruction, keyframes: [Keyframe], lidar: PointCloud,
                      useICP: Bool = true) throws -> AlignmentOutcome {
        guard lidar.count > 1000, photo.cloud.count > 1000 else { throw ProcessingError.noLayer }
        var centers: [Int: SIMD3<Double>] = [:]
        for k in keyframes { centers[k.id] = k.center }
        guard let fit = Alignment.fitFromCameraPoses(photoPoses: photo.poses, arkitCenters: centers)
        else { throw ProcessingError.alignmentFailed }

        var aligned = photo.cloud.transformed(fit.sim)
        let lidarSub = lidar.strided(maxCount: 400_000)

        var r = "Hizalama raporu\n===============\n"
        r += "Anahtar kare: \(keyframes.count), eşleşen poz: \(photo.poses.count)\n"
        r += "LiDAR noktası: \(lidar.count), fotogrametri noktası: \(photo.cloud.count)\n"
        r += String(format: "Poz tabanlı ölçek: %.5f, kamera merkezi RMSE: %.2f cm (%@)\n",
                    fit.sim.s, fit.rmse * 100, fit.usedInverse ? "dünya→kamera" : "kamera→dünya")

        if useICP {
            let icp = Alignment.icp(moving: aligned.strided(maxCount: 30_000), target: lidarSub, estimateScale: true)
            let ok = icp.inlierRatio > 0.30 && icp.sim.s > 0.9 && icp.sim.s < 1.1
            r += String(format: "ICP: %d iter., RMSE %.2f cm, eşleşme %%%.0f, ek ölçek %.5f — %@\n",
                        icp.iterations, icp.rmse * 100, icp.inlierRatio * 100, icp.sim.s, ok ? "uygulandı" : "ATLANDI (güvenilmez)")
            if ok { aligned = aligned.transformed(icp.sim) }
        }
        if let s = Alignment.cloudToCloudStats(photo: aligned.strided(maxCount: 50_000), lidar: lidarSub) {
            r += String(format: "Fotogrametri→LiDAR: ortalama %.2f cm, medyan %.2f cm, RMSE %.2f cm, %%95 %.2f cm\n",
                        s.mean * 100, s.median * 100, s.rmse * 100, s.p95 * 100)
        }
        return AlignmentOutcome(aligned: aligned, report: r)
    }
}
