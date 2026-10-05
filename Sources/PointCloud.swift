import Foundation
import simd

/// Benzerlik dönüşümü (ölçek + dönme + öteleme): p' = s * R * p + t
struct Sim3 {
    var s: Double = 1
    var R: simd_double3x3 = matrix_identity_double3x3
    var t: SIMD3<Double> = .zero

    func apply(_ p: SIMD3<Double>) -> SIMD3<Double> { s * (R * p) + t }

    /// self'ten sonra other uygula: other(self(p))
    func then(_ o: Sim3) -> Sim3 {
        Sim3(s: o.s * s, R: o.R * R, t: o.s * (o.R * t) + o.t)
    }
}

struct VoxelKey: Hashable {
    var x: Int32, y: Int32, z: Int32
    init(_ p: SIMD3<Float>, size: Float) {
        x = Int32((p.x / size).rounded(.down))
        y = Int32((p.y / size).rounded(.down))
        z = Int32((p.z / size).rounded(.down))
    }
    init(x: Int32, y: Int32, z: Int32) { self.x = x; self.y = y; self.z = z }
}

struct PointCloud {
    var positions: [SIMD3<Float>] = []
    var colors: [SIMD3<UInt8>] = []
    var count: Int { positions.count }

    mutating func append(_ p: SIMD3<Float>, _ c: SIMD3<UInt8>) {
        positions.append(p); colors.append(c)
    }

    mutating func append(contentsOf o: PointCloud) {
        positions.append(contentsOf: o.positions)
        colors.append(contentsOf: o.colors)
    }

    func transformed(_ m: Sim3) -> PointCloud {
        var out = self
        for i in 0..<positions.count {
            let p = SIMD3<Double>(positions[i])
            out.positions[i] = SIMD3<Float>(m.apply(p))
        }
        return out
    }

    /// Voksel ızgarası ile seyreltme (her voksel için ortalama).
    func voxelDownsampled(size: Float) -> PointCloud {
        struct Acc { var p = SIMD3<Float>.zero; var c = SIMD3<Float>.zero; var n: Float = 0 }
        var map: [VoxelKey: Acc] = [:]
        map.reserveCapacity(count / 2)
        for i in 0..<count {
            let k = VoxelKey(positions[i], size: size)
            var a = map[k] ?? Acc()
            a.p += positions[i]
            a.c += SIMD3<Float>(colors[i])
            a.n += 1
            map[k] = a
        }
        var out = PointCloud()
        out.positions.reserveCapacity(map.count)
        out.colors.reserveCapacity(map.count)
        for (_, a) in map {
            out.positions.append(a.p / a.n)
            let c = a.c / a.n
            out.colors.append(SIMD3<UInt8>(UInt8(min(255, max(0, c.x))),
                                           UInt8(min(255, max(0, c.y))),
                                           UInt8(min(255, max(0, c.z)))))
        }
        return out
    }

    func strided(maxCount: Int) -> [SIMD3<Float>] {
        guard count > maxCount else { return positions }
        let step = Double(count) / Double(maxCount)
        return (0..<maxCount).map { positions[Int(Double($0) * step)] }
    }
}
