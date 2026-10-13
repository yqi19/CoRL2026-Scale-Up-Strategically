"""
PickCube-v1 is a basic/common task which defaults to using the panda robot. It is also used as a testing task to check whether a robot with manipulation
capabilities can be simulated and trained properly. The configs below set the pick cube task differently to ensure the cube is within reach of the robot tested
and the camera angles are reasonable.
"""
import os

# Object colors: 0=red, 1=green, 2=blue (RGBA; object_color_id for the *Color envs)
OBJECT_COLORS = [
    [1, 0, 0, 1],   # 0: red
    [0, 1, 0, 1],   # 1: green
    [0, 0, 1, 1],   # 2: blue
]

# Verb/Shape tabletop API: 6 colors (object_color_id 0..5)
OBJECT_COLORS_VERB = [
    [1, 0, 0, 1],       # 0: red
    [1, 1, 0, 1],       # 1: yellow
    [0, 0, 1, 1],       # 2: blue
    [1, 0.45, 0, 1],    # 3: orange
    [0, 0.65, 0.2, 1],  # 4: green
    [0.12, 0.12, 0.12, 1],  # 5: black (slightly off pure black for shading)
]

PICK_CUBE_CONFIGS = {
    "panda": {
        "cube_half_size": 0.02,
        "goal_thresh": 0.025,
        "cube_spawn_half_size": 0.1,
        "cube_spawn_center": (0, 0),
        "max_goal_height": 0.3,
        "sensor_cam_eye_pos": [
            0.3,
            0,
            0.6,
        ],  # sensor cam is the camera used for visual observation generation
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [
            0.6,
            0.7,
            0.6,
        ],  # human cam is the camera used for human rendering (i.e. eval videos)
        "human_cam_target_pos": [0.0, 0.0, 0.35],
    },
    "fetch": {
        "cube_half_size": 0.02,
        "goal_thresh": 0.025,
        "cube_spawn_half_size": 0.1,
        "cube_spawn_center": (0, 0),
        "max_goal_height": 0.3,
        "sensor_cam_eye_pos": [0.3, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],
        "human_cam_target_pos": [0.0, 0.0, 0.35],
    },
    "xarm6_robotiq": {
        "cube_half_size": 0.02,
        "goal_thresh": 0.025,
        "cube_spawn_half_size": 0.1,
        "cube_spawn_center": (0, 0),
        "max_goal_height": 0.3,
        "sensor_cam_eye_pos": [0.3, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],
        "human_cam_target_pos": [0.0, 0.0, 0.35],
    },
    "so100": {
        "cube_half_size": 0.0125,
        "goal_thresh": 0.0125 * 1.25,
        "cube_spawn_half_size": 0.05,
        "cube_spawn_center": (-0.46, 0),
        "max_goal_height": 0.08,
        "sensor_cam_eye_pos": [-0.27, 0, 0.4],
        "sensor_cam_target_pos": [-0.56, 0, -0.25],
        "human_cam_eye_pos": [-0.1, 0.3, 0.4],
        "human_cam_target_pos": [-0.46, 0.0, 0.1],
    },
    "widowxai": {
        "cube_half_size": 0.018,
        "goal_thresh": 0.018 * 1.25,
        "cube_spawn_half_size": 0.05,
        "cube_spawn_center": (-0.25, 0),
        "max_goal_height": 0.2,
        "sensor_cam_eye_pos": [0.0, 0, 0.35],
        "sensor_cam_target_pos": [-0.2, 0, 0.1],
        "human_cam_eye_pos": [0.45, 0.5, 0.5],
        "human_cam_target_pos": [-0.2, 0.0, 0.2],
    },
    "panda_wristcam": {
        "cube_half_size": 0.02,
        "goal_thresh": 0.025,
        "cube_spawn_half_size": 0.1,
        "cube_spawn_center": (0, 0),
        "max_goal_height": 0.3,
        "radius": 0.03,
        "ring_radius": 0.03,
        "ring_inner_radius": 0.015,
        "ring_half_height": 0.008,
        "sensor_cam_eye_pos": [0.3, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],
        "human_cam_target_pos": [0.0, 0.0, 0.35],
    },
}

# Workspace (all Verb*Color envs): x,y in [-0.1, 0.1] m
WORKSPACE_SPAWN_HALF_SIZE = 0.1
WORKSPACE_SPAWN_CENTER = (0, 0)

# Default object sizes file (cube, sphere, ring). Envs can override via object_sizes_path or object_sizes kwargs.
OBJECT_SIZES_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "object_sizes_default.txt")


def load_object_sizes(path: str = None) -> dict:
    """Load object sizes from a .txt file (key=value per line, # comments). Returns dict with float values.
    Keys: cube_half_size, sphere_radius, ring_radius, ring_inner_radius, ring_half_height.
    If path is None, uses OBJECT_SIZES_DEFAULT_PATH.
    """
    if path is None:
        path = OBJECT_SIZES_DEFAULT_PATH
    out = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                try:
                    out[k] = float(v)
                except ValueError:
                    pass
    return out


def apply_object_sizes_to_cfg(cfg: dict, object_sizes_path: str = None, object_sizes: dict = None) -> None:
    """Update cfg in-place with sizes from file or dict. Keys only overwritten if present in cfg.
    File/dict may use sphere_radius (mapped to cfg['radius']).
    """
    def _apply(src: dict) -> None:
        if "cube_half_size" in src and "cube_half_size" in cfg:
            cfg["cube_half_size"] = src["cube_half_size"]
        if "radius" in src and "radius" in cfg:
            cfg["radius"] = src["radius"]
        if "sphere_radius" in src and "radius" in cfg:
            cfg["radius"] = src["sphere_radius"]
        for k in ("ring_radius", "ring_inner_radius", "ring_half_height"):
            if k in src and k in cfg:
                cfg[k] = src[k]
    if object_sizes_path is not None:
        _apply(load_object_sizes(object_sizes_path))
    if object_sizes is not None:
        _apply(object_sizes)
