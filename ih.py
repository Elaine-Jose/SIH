#!/usr/bin/env python3
"""
Compute geometry, volume, self-weight, slenderness and middle-third check
of a structural element from a point cloud using Open3D.

Assumptions:
- Z axis is vertical (up).
- Input is a single "part" (e.g., one column or one wall).
- Units are meters.
- Loads: P in kN, M in kN·m (for middle-third rule).
"""

import sys
import math
import numpy as np
import open3d as o3d


# ---- CONFIG (tune as you like) ----
VOXEL_SIZE = 0.02        # [m] 2 cm downsampling
SLICE_THICKNESS = 0.10   # [m] thickness of horizontal slice at mid-height
DENSITY_CONCRETE = 25_000.0  # [N/m^3] ~25 kN/m^3
DEFAULT_SLENDER_LIMIT = 60.0  # Typical RC column limit (adjust to your code)


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"No points found in file: {path}")
    return pcd


def downsample(pcd: o3d.geometry.PointCloud,
               voxel_size: float) -> o3d.geometry.PointCloud:
    if voxel_size is None or voxel_size <= 0:
        return pcd
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def compute_height(points: np.ndarray):
    z_min = points[:, 2].min()
    z_max = points[:, 2].max()
    return float(z_max - z_min), float(z_min), float(z_max)


def extract_mid_slice(points: np.ndarray,
                      z_min: float,
                      z_max: float,
                      slice_thickness: float) -> np.ndarray:
    """
    Take a horizontal slice around mid-height and return XY points.
    If too few points in the slice, fallback to all XY points.
    """
    mid_z = 0.5 * (z_min + z_max)
    half_thick = 0.5 * slice_thickness

    mask = (points[:, 2] >= mid_z - half_thick) & \
           (points[:, 2] <= mid_z + half_thick)

    slice_pts = points[mask]

    # Fallback if the slice is too sparse
    if slice_pts.shape[0] < 50:   # threshold; adjust as needed
        # print("Warning: too few points in mid slice, using all points for XY extent.")
        slice_pts = points

    # Return only XY
    return slice_pts[:, :2]


def fit_2d_oriented_bbox(xy_points: np.ndarray):
    """
    Fit an oriented bounding box in 2D using Open3D by embedding in 3D (z=0).
    Returns (B, D) as the two in-plane extents.
    """
    if xy_points.shape[0] < 3:
        raise ValueError("Not enough points to fit bounding box.")

    # Build a fake 3D point cloud at z=0
    pts_3d = np.zeros((xy_points.shape[0], 3), dtype=np.float64)
    pts_3d[:, 0:2] = xy_points

    pcd_slice = o3d.geometry.PointCloud()
    pcd_slice.points = o3d.utility.Vector3dVector(pts_3d)

    obb = pcd_slice.get_oriented_bounding_box()
    ex, ey, ez = obb.extent  # ez should be very small (~0)

    # B and D are the two in-plane dimensions
    # Sort to be consistent: B = larger, D = smaller
    B, D = sorted([ex, ey], reverse=True)
    return float(B), float(D)


def slenderness_check(height: float,
                      B: float,
                      D: float,
                      area: float,
                      slender_limit: float):
    """
    Compute slenderness ratio using the weaker axis (smaller dimension D).
    Uses:
        I = B * D^3 / 12
        r = sqrt(I / A)
        lambda = Le / r   with Le ≈ height (can be refined later).
    """
    # Moment of inertia about the "weak" axis (through centroid, bending about B)
    I_weak = B * (D ** 3) / 12.0  # m^4
    r_weak = math.sqrt(I_weak / area)  # m

    # For now, take effective length Le = actual height
    Le = height

    slenderness = Le / r_weak if r_weak > 0 else float("inf")

    is_safe = slenderness <= slender_limit

    return {
        "I_weak": I_weak,
        "r_weak": r_weak,
        "Le": Le,
        "slenderness": slenderness,
        "slender_limit": slender_limit,
        "is_safe": is_safe,
    }


def middle_third_check(B: float,
                       P_kN: float,
                       M_kNm: float):
    """
    Middle third rule:
        e = M / P
        safe if e <= B/6

    Units:
    - P in kN
    - M in kN·m
    => e in meters.
    """
    if P_kN <= 0.0:
        raise ValueError("Axial load P must be > 0 for middle-third check.")

    e = M_kNm / P_kN  # [m]
    e_limit = B / 6.0

    is_safe = e <= e_limit

    return {
        "e": e,
        "e_limit": e_limit,
        "is_safe": is_safe,
    }


def main(path: str,
         P_kN: float | None = None,
         M_kNm: float | None = None,
         slender_limit: float = DEFAULT_SLENDER_LIMIT):

    print(f"Loading point cloud: {path}")
    pcd = load_point_cloud(path)

    print(f"Original points: {len(pcd.points)}")
    pcd = downsample(pcd, VOXEL_SIZE)
    print(f"Downsampled points: {len(pcd.points)}")

    points = np.asarray(pcd.points)
    height, z_min, z_max = compute_height(points)
    print(f"Z min = {z_min:.3f} m, Z max = {z_max:.3f} m")
    print(f"Estimated height (L) = {height:.3f} m")

    # Mid-height slice for cross-section
    xy_slice = extract_mid_slice(points, z_min, z_max, SLICE_THICKNESS)

    # Fit 2D oriented bounding box -> B & D
    B, D = fit_2d_oriented_bbox(xy_slice)
    print(f"Estimated cross-section dimensions at mid-height:")
    print(f"  B (larger dim) = {B:.3f} m")
    print(f"  D (smaller dim) = {D:.3f} m")

    # Area, volume, weight
    area = B * D
    volume = area * height
    weight_N = volume * DENSITY_CONCRETE
    weight_kN = weight_N / 1000.0

    print("\n--- Geometry & Weight ---")
    print(f"Area A = {area:.4f} m²")
    print(f"Volume V = {volume:.4f} m³")
    print(f"Self-weight ≈ {weight_kN:.2f} kN "
          f"(assuming density = {DENSITY_CONCRETE/1000:.1f} kN/m³)")

    # ---- Slenderness check ----
    slender = slenderness_check(height, B, D, area, slender_limit)

    print("\n--- Slenderness Check (using weaker axis) ---")
    print(f"I_weak = {slender['I_weak']:.6f} m⁴")
    print(f"r_weak = {slender['r_weak']:.4f} m")
    print(f"Effective length Le ≈ {slender['Le']:.3f} m")
    print(f"Slenderness λ = {slender['slenderness']:.2f}")
    print(f"Limit λ_max = {slender['slender_limit']:.1f}")

    if slender["is_safe"]:
        print("Result: ✅ SAFE in slenderness (λ ≤ limit)")
    else:
        print("Result: ❌ NOT SAFE in slenderness (λ > limit)")

    # ---- Middle third rule (if P and M are provided) ----
    if P_kN is not None and M_kNm is not None:
        m3 = middle_third_check(B, P_kN, M_kNm)

        print("\n--- Middle Third Rule Check ---")
        print(f"Axial load P = {P_kN:.2f} kN")
        print(f"Moment M = {M_kNm:.2f} kN·m")
        print(f"Eccentricity e = {m3['e']:.4f} m")
        print(f"Limit e_max = B/6 = {m3['e_limit']:.4f} m")

        if m3["is_safe"]:
            print("Result: ✅ SAFE by middle third rule (e ≤ B/6, no tension)")
        else:
            print("Result: ❌ NOT SAFE by middle third rule (e > B/6, tension develops)")
    else:
        print("\nMiddle third check skipped (P_kN and/or M_kNm not provided).")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python compute_volume_from_pointcloud.py "
              "<pointcloud.ply> [P_kN] [M_kNm] [slender_limit]")
        print("Examples:")
        print("  python compute_volume_from_pointcloud.py col1.ply")
        print("  python compute_volume_from_pointcloud.py col1.ply 800 120 60")
        sys.exit(1)

    path = sys.argv[1]

    P_kN_arg = None
    M_kNm_arg = None
    slender_limit_arg = DEFAULT_SLENDER_LIMIT

    if len(sys.argv) >= 3:
        P_kN_arg = float(sys.argv[2])
    if len(sys.argv) >= 4:
        M_kNm_arg = float(sys.argv[3])
    if len(sys.argv) >= 5:
        slender_limit_arg = float(sys.argv[4])

    main(path, P_kN_arg, M_kNm_arg, slender_limit_arg)
