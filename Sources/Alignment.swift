import Foundation
import simd

/// Fotogrametri nokta bulutunu LiDAR (ARKit) ölçeğine ve koordinat sistemine oturtur.
/// 1) Kamera merkezleri üzerinden Umeyama/Horn benzerlik dönüşümü (ölçek dahil)
/// 2) LiDAR bulutuna karşı kırpılmış (trimmed) ölçek-duyarlı ICP ile ince ayar
enum Alignment {

    // MARK: Horn (1987) kapalı biçim benzerlik dönüşümü

    /// src -> dst eşleşmeleri için en iyi s, R, t. (en az 3 doğrusal olmayan nokta)
    static func similarity(src: [SIMD3<Double>], dst: [SIMD3<Double>], estimateScale: Bool = true) -> Sim3? {
        let n = src.count
        guard n >= 3, dst.count == n else { return nil }
        let cs = src.reduce(.zero, +) / Double(n)
        let cd = dst.reduce(.zero, +) / Double(n)

        var Sxx = 0.0, Sxy = 0.0, Sxz = 0.0, Syx = 0.0, Syy = 0.0, Syz = 0.0, Szx = 0.0, Szy = 0.0, Szz = 0.0
        var srcVar = 0.0
        for i in 0..<n {
            let a = src[i] - cs, b = dst[i] - cd
            Sxx += a.x * b.x; Sxy += a.x * b.y; Sxz += a.x * b.z
            Syx += a.y * b.x; Syy += a.y * b.y; Syz += a.y * b.z
            Szx += a.z * b.x; Szy += a.z * b.y; Szz += a.z * b.z
            srcVar += simd_length_squared(a)
        }
        guard srcVar > 1e-12 else { return nil }

        var N: [[Double]] = [
            [Sxx + Syy + Szz, Syz - Szy,        Szx - Sxz,        Sxy - Syx],
            [Syz - Szy,       Sxx - Syy - Szz,  Sxy + Syx,        Szx + Sxz],
            [Szx - Sxz,       Sxy + Syx,       -Sxx + Syy - Szz,  Syz + Szy],
            [Sxy - Syx,       Szx + Sxz,        Syz + Szy,       -Sxx - Syy + Szz]
        ]
        let q = largestEigenvector(&N)           // (w, x, y, z)
        let R = rotationMatrix(w: q[0], x: q[1], y: q[2], z: q[3])

        var s = 1.0
        if estimateScale {
            var num = 0.0
            for i in 0..<n { num += simd_dot(dst[i] - cd, R * (src[i] - cs)) }
            s = num / srcVar
            if !(s > 1e-9) { return nil }
        }
        let t = cd - s * (R * cs)
        return Sim3(s: s, R: R, t: t)
    }

    static func rotationMatrix(w: Double, x: Double, y: Double, z: Double) -> simd_double3x3 {
        let n = (w*w + x*x + y*y + z*z).squareRoot()
        let w = w/n, x = x/n, y = y/n, z = z/n
        // satır-ana (row-major) değerler; simd sütun-ana olduğundan sütunlar veriliyor
        let r00 = 1 - 2*(y*y + z*z), r01 = 2*(x*y - w*z),     r02 = 2*(x*z + w*y)
        let r10 = 2*(x*y + w*z),     r11 = 1 - 2*(x*x + z*z), r12 = 2*(y*z - w*x)
        let r20 = 2*(x*z - w*y),     r21 = 2*(y*z + w*x),     r22 = 1 - 2*(x*x + y*y)
        return simd_double3x3(columns: (SIMD3(r00, r10, r20), SIMD3(r01, r11, r21), SIMD3(r02, r12, r22)))
    }

    /// Simetrik 4x4 matrisin en büyük özdeğerine ait özvektör (Jacobi yöntemi).
    static func largestEigenvector(_ A: inout [[Double]]) -> [Double] {
        let n = 4
        var V = (0..<n).map { i in (0..<n).map { $0 == i ? 1.0 : 0.0 } }
        for _ in 0..<60 {
            var off = 0.0
            for i in 0..<n { for j in (i+1)..<n { off += A[i][j] * A[i][j] } }
            if off < 1e-24 { break }
            for p in 0..<n-1 {
                for q in (p+1)..<n {
                    if abs(A[p][q]) < 1e-300 { continue }
                    let theta = (A[q][q] - A[p][p]) / (2 * A[p][q])
                    let t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + (theta*theta + 1).squareRoot())
                    let c = 1 / (t*t + 1).squareRoot(), s = t * c
                    for k in 0..<n {
                        let akp = A[k][p], akq = A[k][q]
                        A[k][p] = c*akp - s*akq
                        A[k][q] = s*akp + c*akq
                    }
                    for k in 0..<n {
                        let apk = A[p][k], aqk = A[q][k]
                        A[p][k] = c*apk - s*aqk
                        A[q][k] = s*apk + c*aqk
                    }
                    for k in 0..<n {
                        let vkp = V[k][p], vkq = V[k][q]
                        V[k][p] = c*vkp - s*vkq
                        V[k][q] = s*vkp + c*vkq
                    }
                }
            }
        }
        var best = 0
        for i in 1..<n where A[i][i] > A[best][best] { best = i }
        return (0..<n).map { V[$0][best] }
    }

    // MARK: Kamera merkezlerinden ilk hizalama

    struct CenterFit { var sim: Sim3; var rmse: Double; var usedInverse: Bool }

    /// Object Capture pozlarının yönü (cam→dünya / dünya→cam) belgelerde net olmadığından
    /// iki hipotezi de dener, kalıntısı küçük olanı seçer.
    static func fitFromCameraPoses(photoPoses: [Int: simd_float4x4],
                                   arkitCenters: [Int: SIMD3<Double>]) -> CenterFit? {
        let ids = photoPoses.keys.filter { arkitCenters[$0] != nil }.sorted()
        guard ids.count >= 4 else { return nil }
        let dst = ids.map { arkitCenters[$0]! }

        func fit(_ centers: [SIMD3<Double>], inverse: Bool) -> CenterFit? {
            guard let m = similarity(src: centers, dst: dst) else { return nil }
            var e = 0.0
            for i in 0..<ids.count { e += simd_length_squared(m.apply(centers[i]) - dst[i]) }
            return CenterFit(sim: m, rmse: (e / Double(ids.count)).squareRoot(), usedInverse: inverse)
        }
        let direct = ids.map { id -> SIMD3<Double> in
            let c = photoPoses[id]!.columns.3
            return SIMD3<Double>(Double(c.x), Double(c.y), Double(c.z))
        }
        let inverse = ids.map { id -> SIMD3<Double> in
            let m = photoPoses[id]!
            let R = simd_float3x3(SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z),
                                  SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z),
                                  SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z))
            let t = SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z)
            let c = -(R.transpose * t)
            return SIMD3<Double>(c)
        }
        let a = fit(direct, inverse: false), b = fit(inverse, inverse: true)
        switch (a, b) {
        case let (x?, y?): return x.rmse <= y.rmse ? x : y
        case let (x?, nil): return x
        case let (nil, y?): return y
        default: return nil
        }
    }

    // MARK: ICP

    struct ICPResult { var sim: Sim3; var rmse: Double; var inlierRatio: Double; var iterations: Int }

    /// `moving` (fotogrametri, zaten kaba hizalanmış) → `target` (LiDAR).
    /// Dönen Sim3, moving'e uygulanacak ek dönüşümdür.
    static func icp(moving: [SIMD3<Float>], target: [SIMD3<Float>],
                    estimateScale: Bool = true, maxIterations: Int = 40) -> ICPResult {
        let cell: Float = 0.10
        var grid: [VoxelKey: [Int32]] = [:]
        for (i, p) in target.enumerated() { grid[VoxelKey(p, size: cell), default: []].append(Int32(i)) }

        var cur = moving.map { SIMD3<Double>($0) }
        var total = Sim3()
        var lastRMSE = Double.infinity
        var inlier = 0.0
        var iters = 0
        let thresholds: [Double] = [0.10, 0.08, 0.06, 0.05, 0.04, 0.03]

        for it in 0..<maxIterations {
            iters = it + 1
            let thr = thresholds[min(it, thresholds.count - 1)]
            // En yakın komşu araması tüm çekirdeklerde paralel
            let m = cur.count
            var nn = [Int32](repeating: -1, count: m)
            var nd = [Double](repeating: .infinity, count: m)
            let chunk = 2048, chunks = (m + chunk - 1) / chunk
            nn.withUnsafeMutableBufferPointer { nnb in
                nd.withUnsafeMutableBufferPointer { ndb in
                    let nnp = nnb.baseAddress!, ndp = ndb.baseAddress!
                    cur.withUnsafeBufferPointer { curb in
                        DispatchQueue.concurrentPerform(iterations: chunks) { ch in
                            for i in (ch * chunk)..<min(m, (ch + 1) * chunk) {
                                let pf = SIMD3<Float>(curb[i])
                                let k = VoxelKey(pf, size: cell)
                                var bestD = Float.infinity
                                var bestI: Int32 = -1
                                for dx in Int32(-1)...1 { for dy in Int32(-1)...1 { for dz in Int32(-1)...1 {
                                    guard let list = grid[VoxelKey(x: k.x + dx, y: k.y + dy, z: k.z + dz)] else { continue }
                                    for idx in list {
                                        let d = simd_length_squared(target[Int(idx)] - pf)
                                        if d < bestD { bestD = d; bestI = idx }
                                    }
                                }}}
                                nnp[i] = bestI; ndp[i] = Double(bestD)
                            }
                        }
                    }
                }
            }
            var src: [SIMD3<Double>] = [], dst: [SIMD3<Double>] = [], d2: [Double] = []
            for i in 0..<m where nn[i] >= 0 && nd[i].squareRoot() < thr {
                src.append(cur[i]); dst.append(SIMD3<Double>(target[Int(nn[i])])); d2.append(nd[i])
            }
            inlier = Double(src.count) / Double(max(1, cur.count))
            guard src.count >= 100 else { break }

            // en iyi %80'i tut (kırpılmış ICP)
            let order = d2.indices.sorted { d2[$0] < d2[$1] }
            let keep = Array(order.prefix(Int(Double(order.count) * 0.8)))
            let s2 = keep.map { src[$0] }, d3 = keep.map { dst[$0] }
            let rmse = (keep.reduce(0.0) { $0 + d2[$1] } / Double(keep.count)).squareRoot()

            guard let delta = similarity(src: s2, dst: d3, estimateScale: estimateScale) else { break }
            for i in 0..<cur.count { cur[i] = delta.apply(cur[i]) }
            total = total.then(delta)
            if abs(lastRMSE - rmse) < 1e-5 { lastRMSE = rmse; break }
            lastRMSE = rmse
        }
        return ICPResult(sim: total, rmse: lastRMSE, inlierRatio: inlier, iterations: iters)
    }

    /// Hizalama sonrası doğruluk: her fotogrametri noktasından en yakın LiDAR noktasına uzaklık.
    static func cloudToCloudStats(photo: [SIMD3<Float>], lidar: [SIMD3<Float>]) -> (mean: Double, rmse: Double, median: Double, p95: Double)? {
        let cell: Float = 0.10
        var grid: [VoxelKey: [Int32]] = [:]
        for (i, p) in lidar.enumerated() { grid[VoxelKey(p, size: cell), default: []].append(Int32(i)) }
        var ds: [Double] = []
        for p in photo {
            let k = VoxelKey(p, size: cell)
            var best = Float.infinity
            for dx in Int32(-1)...1 { for dy in Int32(-1)...1 { for dz in Int32(-1)...1 {
                guard let l = grid[VoxelKey(x: k.x + dx, y: k.y + dy, z: k.z + dz)] else { continue }
                for i in l { best = min(best, simd_length_squared(lidar[Int(i)] - p)) }
            }}}
            if best.isFinite { ds.append(Double(best).squareRoot()) }
        }
        guard ds.count > 10 else { return nil }
        ds.sort()
        let mean = ds.reduce(0, +) / Double(ds.count)
        let rmse = (ds.reduce(0) { $0 + $1 * $1 } / Double(ds.count)).squareRoot()
        return (mean, rmse, ds[ds.count / 2], ds[Int(Double(ds.count) * 0.95)])
    }
}
