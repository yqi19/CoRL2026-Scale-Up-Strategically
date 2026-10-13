"""
Tabletop objects: cube, sphere, cup, ring, car (OBJ), pyramid, star (procedural prism).

``cup`` / ``car`` load from ``asset/cup.obj`` and ``asset/car.obj`` when present;
``cup`` falls back to a procedural mug if the mesh fails.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import trimesh
import trimesh.transformations as tft
from transforms3d.euler import euler2quat

from mani_skill import logger
from mani_skill.envs.scene import ManiSkillScene

# (filename, target max AABB extent m, strip outliers, euler_xyz)
_MESH_SPECS: dict[str, tuple[str, float, bool, tuple[float, float, float]]] = {
    "cup": ("cup.obj", 0.046, True, (np.pi / 2, 0.0, 0.0)),
    "car": ("car.obj", 0.096, False, (np.pi / 2, 0.0, 0.0)),
}


def _mat(color_rgba):
    m = sapien.render.RenderMaterial()
    m.set_base_color(list(color_rgba))
    return m


def _vis_mat(color_rgba):
    return sapien.render.RenderMaterial(
        base_color=list(color_rgba),
        roughness=0.38,
        specular=0.25,
    )


def _asset_obj_dir() -> Path | None:
    here = Path(__file__).resolve()
    for anc in here.parents:
        d = anc / "asset"
        if d.is_dir() and (d / "cup.obj").is_file() and (d / "car.obj").is_file():
            return d
    return None


def _export_tabletop_trimesh(
    m: trimesh.Trimesh,
    *,
    target_max_extent: float,
    euler_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[str, float]:
    m = m.copy()
    if euler_xyz != (0.0, 0.0, 0.0):
        T = tft.euler_matrix(euler_xyz[0], euler_xyz[1], euler_xyz[2], axes="sxyz")
        m.apply_transform(T)
    ext = float(m.extents.max())
    if ext < 1e-9:
        raise RuntimeError("degenerate mesh extent")
    m.apply_scale(float(target_max_extent) / ext)
    cxy = (m.bounds[0, :2] + m.bounds[1, :2]) / 2.0
    m.vertices[:, 0] -= cxy[0]
    m.vertices[:, 1] -= cxy[1]
    z0, z1 = float(m.bounds[0, 2]), float(m.bounds[1, 2])
    zc = (z0 + z1) / 2.0
    m.vertices[:, 2] -= zc
    rest_z = (z1 - z0) / 2.0

    # Avoid exporting texture-backed materials that can reference ephemeral files
    # like /tmp/material_0.png (racey under parallel collection).
    try:
        m.visual = m.visual.to_color()
    except Exception:
        pass

    fd, tmp_path = tempfile.mkstemp(suffix=".obj", prefix="ms_tabletop_mesh_")
    os.close(fd)
    m.export(tmp_path)
    return tmp_path, rest_z


def _prepare_tabletop_mesh(
    src: Path,
    *,
    target_max_extent: float,
    euler_xyz: tuple[float, float, float],
    strip_cup_outliers: bool,
) -> tuple[str, float]:
    m = trimesh.load(src, force="mesh", skip_materials=True)
    m = m.copy()
    if strip_cup_outliers:
        keep_v = (np.abs(m.vertices[:, 0]) < 5.0) & (np.abs(m.vertices[:, 2]) < 5.0)
        face_ok = keep_v[m.faces].all(axis=1)
        m.update_faces(face_ok)
        m.remove_unreferenced_vertices()
    if len(m.faces) == 0:
        raise RuntimeError("no faces left after mesh cleanup")
    if euler_xyz != (0.0, 0.0, 0.0):
        T = tft.euler_matrix(euler_xyz[0], euler_xyz[1], euler_xyz[2], axes="sxyz")
        m.apply_transform(T)
    ext = float(m.extents.max())
    if ext < 1e-9:
        raise RuntimeError("degenerate mesh extent")
    m.apply_scale(float(target_max_extent) / ext)
    cxy = (m.bounds[0, :2] + m.bounds[1, :2]) / 2.0
    m.vertices[:, 0] -= cxy[0]
    m.vertices[:, 1] -= cxy[1]
    z0, z1 = float(m.bounds[0, 2]), float(m.bounds[1, 2])
    zc = (z0 + z1) / 2.0
    m.vertices[:, 2] -= zc
    rest_z = (z1 - z0) / 2.0

    # Force color visuals to avoid exporting external texture references.
    try:
        m.visual = m.visual.to_color()
    except Exception:
        pass

    fd, tmp_path = tempfile.mkstemp(suffix=".obj", prefix="ms_tabletop_mesh_")
    os.close(fd)
    m.export(tmp_path)
    return tmp_path, rest_z


def _cook_mesh_convex_visual(
    b,
    tmp_path: str,
    color_rgba,
    *,
    static_friction: float = 0.45,
    dynamic_friction: float = 0.45,
) -> bool:
    pmat = physx.PhysxMaterial(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=0.0,
    )
    try:
        b.add_convex_collision_from_file(
            tmp_path, scale=(1, 1, 1), material=pmat, density=400.0
        )
        b.add_visual_from_file(tmp_path, scale=(1, 1, 1), material=_vis_mat(color_rgba))
        return True
    except Exception as e:
        logger.warning("tabletop mesh cook failed (%s)", e)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return False


def _try_mesh_shape(
    b, shape: str, color_rgba, size_scale: float
) -> tuple[float | None, str | None]:
    spec = _MESH_SPECS.get(shape)
    if spec is None:
        return None, None
    adir = _asset_obj_dir()
    if adir is None:
        return None, None
    fname, target_ext, strip, euler = spec
    src = adir / fname
    if not src.is_file():
        return None, None
    try:
        tmp_path, rest_z = _prepare_tabletop_mesh(
            src,
            target_max_extent=float(target_ext) * float(size_scale),
            euler_xyz=euler,
            strip_cup_outliers=strip,
        )
    except Exception as e:
        logger.warning("tabletop mesh %s failed (%s)", src, e)
        return None, None
    if not _cook_mesh_convex_visual(b, tmp_path, color_rgba):
        return None, None
    return float(rest_z), tmp_path


def _star_polygon_verts_faces(outer_r: float, inner_r: float, n: int = 5):
    tips, valleys = [], []
    for i in range(n):
        at = -np.pi / 2 + i * 2 * np.pi / n
        tips.append([outer_r * np.cos(at), outer_r * np.sin(at)])
        av = -np.pi / 2 + (i + 0.5) * 2 * np.pi / n
        valleys.append([inner_r * np.cos(av), inner_r * np.sin(av)])
    verts = []
    for i in range(n):
        verts.append(tips[i])
        verts.append(valleys[i])
    verts = np.array(verts, dtype=np.float64)
    faces = []
    for i in range(n):
        tip = 2 * i
        v_prev = 2 * ((i - 1) % n) + 1
        v_curr = 2 * i + 1
        faces.append([tip, v_prev, v_curr])
    inner_idx = [2 * i + 1 for i in range(n)]
    c = verts[inner_idx].mean(axis=0)
    c_idx = len(verts)
    verts = np.vstack([verts, c])
    for i in range(n):
        faces.append([c_idx, inner_idx[i], inner_idx[(i + 1) % n]])
    return verts, np.array(faces, dtype=np.int64)


def _make_star_prism_trimesh(s: float) -> trimesh.Trimesh:
    R = 0.024 * s
    r = 0.010 * s
    h = 0.016 * s
    v2, f2 = _star_polygon_verts_faces(R, r, n=5)
    star = trimesh.creation.extrude_triangulation(v2, f2, height=h, process=False)
    r_plug = min(r * 0.92, R * 0.42)
    plug = trimesh.creation.cylinder(radius=r_plug, height=h, sections=28)
    plug.apply_translation([0.0, 0.0, h * 0.5])
    return trimesh.util.concatenate([star, plug])


def _make_square_pyramid_trimesh(s: float) -> trimesh.Trimesh:
    b = 0.019 * s
    h = 0.036 * s
    verts = np.array(
        [
            [-b, -b, 0.0],
            [b, -b, 0.0],
            [b, b, 0.0],
            [-b, b, 0.0],
            [0.0, 0.0, h],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def build_tabletop_object(
    scene: ManiSkillScene,
    shape: str,
    color_rgba,
    name: str = "obj",
    *,
    size_scale: float = 1.0,
):
    """
    ``shape`` in cube | sphere | cup | ring | car | pyramid | star.
    Aliases: ``can``, ``mug``, ``cylinder`` → ``cup``.
    """
    shape = shape.lower().strip()
    if shape in ("can", "mug", "cylinder"):
        shape = "cup"
    elif shape == "bottle":
        raise ValueError(
            "shape 'bottle' is not supported; use: cube, sphere, cup, ring, car, pyramid, star"
        )
    s = max(float(size_scale), 1e-6)
    mat = _mat(color_rgba)
    b = scene.create_actor_builder()
    tmp_mesh: str | None = None
    rest_z: float

    if shape in ("cup", "car"):
        rz_mesh, tmp_mesh = _try_mesh_shape(b, shape, color_rgba, s)
        if rz_mesh is not None:
            rest_z = rz_mesh
        elif shape == "cup":
            b = scene.create_actor_builder()
            tmp_mesh = None
            # Procedural mug (slightly smaller base than earlier 0.024 / 0.04 m scale).
            R_out = 0.020 * s
            R_in = 0.013 * s
            H = 0.034 * s
            tb = 0.004 * s
            Hh = H / 2
            n_seg = 12
            r_mid = (R_out + R_in) * 0.5
            wall_rad = (R_out - R_in) * 0.5
            tan_half = R_out * np.sin(np.pi / n_seg) * 0.92
            z_bot_disc = -Hh + tb / 2
            b.add_cylinder_collision(
                pose=sapien.Pose(p=[0, 0, z_bot_disc]),
                radius=R_out,
                half_length=tb / 2,
            )
            b.add_cylinder_visual(
                pose=sapien.Pose(p=[0, 0, z_bot_disc]),
                radius=R_out,
                half_length=tb / 2,
                material=mat,
            )
            z_low = -Hh + tb
            z_high = Hh
            z_wall = (z_low + z_high) / 2
            hh_wall = (z_high - z_low) / 2
            for i in range(n_seg):
                ang = (i + 0.5) * (2 * np.pi / n_seg)
                px = r_mid * np.cos(ang)
                py = r_mid * np.sin(ang)
                q = euler2quat(0, 0, ang)
                pose = sapien.Pose(p=[px, py, z_wall], q=q)
                b.add_box_collision(
                    pose=pose, half_size=[wall_rad, tan_half, hh_wall]
                )
                b.add_box_visual(
                    pose=pose, half_size=[wall_rad, tan_half, hh_wall], material=mat
                )
            hx, hy, hz = 0.005 * s, 0.017 * s, 0.015 * s
            ph = sapien.Pose(p=[R_out + hx * 0.85 + 0.002 * s, 0, 0.003 * s])
            b.add_box_collision(pose=ph, half_size=[hx, hy, hz])
            b.add_box_visual(pose=ph, half_size=[hx, hy, hz], material=mat)
            rest_z = Hh
        else:
            raise RuntimeError(
                "car requires asset/car.obj under a repo ``asset/`` directory (or fix the mesh path)."
            )
    elif shape == "star":
        mesh = _make_star_prism_trimesh(s)
        target = 0.05 * s
        tmp_mesh, rest_z = _export_tabletop_trimesh(mesh, target_max_extent=target)
        # Higher friction: tapering meshes slip in grip during rotate; μ helps tangential slip only.
        if not _cook_mesh_convex_visual(
            b, tmp_mesh, color_rgba, static_friction=0.95, dynamic_friction=0.95
        ):
            raise RuntimeError("failed to build star mesh actor")
    elif shape == "pyramid":
        mesh = _make_square_pyramid_trimesh(s)
        target = 0.043 * s
        tmp_mesh, rest_z = _export_tabletop_trimesh(mesh, target_max_extent=target)
        if not _cook_mesh_convex_visual(
            b, tmp_mesh, color_rgba, static_friction=0.95, dynamic_friction=0.95
        ):
            raise RuntimeError("failed to build pyramid mesh actor")
    elif shape == "cube":
        hs = 0.021 * s
        b.add_box_collision(half_size=[hs, hs, hs])
        b.add_box_visual(half_size=[hs, hs, hs], material=mat)
        rest_z = hs
    elif shape == "sphere":
        r = 0.024 * s
        b.add_sphere_collision(radius=r)
        b.add_sphere_visual(radius=r, material=mat)
        rest_z = r
    elif shape == "ring":
        radius = 0.03 * s
        inner_radius = 0.015 * s
        half_height = 0.008 * s
        n_seg = 16
        mid_r = (radius + inner_radius) * 0.5
        seg_r = (radius - inner_radius) * 0.5
        for i in range(n_seg):
            ang = i * (2 * np.pi / n_seg)
            px = mid_r * np.cos(ang)
            py = mid_r * np.sin(ang)
            pose = sapien.Pose(p=[px, py, 0])
            b.add_cylinder_collision(pose=pose, radius=seg_r, half_length=half_height)
            b.add_cylinder_visual(
                pose=pose, radius=seg_r, half_length=half_height, material=mat
            )
        rest_z = half_height
    else:
        raise ValueError(f"Unknown tabletop object shape: {shape!r}")

    b.initial_pose = sapien.Pose(p=[0, 0, float(rest_z)])
    actor = b.build(name=name)
    if tmp_mesh is not None:
        try:
            os.unlink(tmp_mesh)
        except OSError:
            pass
    return actor, float(rest_z)
