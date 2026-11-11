"""GRITS-style trajectory and disturbance overlays from recorded poses.

The public GRITS demo draws two end-effector traces on a camera image and
holds a score bar beside them. This renderer does the same job for CANDID
records: project TCP paths into the scene camera, draw distractor motion,
and report the disturbance score already defined in ``metrics.py``.

A trajectory archive is the ``trajectory.npz`` written by evaluation: TCP
pose, target pose, goal, and the distractor trace. One archive is one
episode. A second episode can be passed beside it, or stored in the same
archive as arrays whose first axis has length 2.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from candid.data.metrics import disturbance_from_poses

SCENE1_CAMERA_EYE = (0.52, 0.65, 0.58)
SCENE1_CAMERA_TARGET = (-0.06, 0.0, 0.12)
# Cyan to blue, then yellow to red. Same pairing as the GRITS legend.
TRACE_COLORS = (
    ((90, 220, 255), (30, 70, 255)),
    ((255, 220, 40), (230, 40, 40)),
)
DISTRACTOR_COLORS = np.array(
    [
        [166, 102, 230],
        [51, 191, 89],
        [242, 158, 51],
        [51, 153, 255],
        [242, 204, 64],
        [235, 110, 133],
    ],
    dtype=np.float64,
)
_BOX_CORNERS = np.array(
    [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
    dtype=np.float64,
)
_BOX_FACES = (
    (0, 1, 3, 2),
    (4, 6, 7, 5),
    (0, 4, 5, 1),
    (2, 3, 7, 6),
    (0, 2, 6, 4),
    (1, 5, 7, 3),
)


def load_trajectories(path, second_path=None):
    """Return one or two trajectory dicts.

    A single evaluation archive stays one episode. Two episodes are accepted
    either as a second file or as arrays stacked on axis 0 with length 2.
    """
    first = _read_archive(path)
    if second_path is not None:
        return [first, _read_archive(second_path)]
    stacked = _split_stacked(first)
    if stacked is not None:
        return stacked
    return [first]


def render_trajectory_video(
    source,
    output_dir,
    second_source=None,
    *,
    eye=SCENE1_CAMERA_EYE,
    target=SCENE1_CAMERA_TARGET,
    width=640,
    height=640,
    fovy=1.0,
    fps=20,
    labels=None,
):
    """Write an mp4, a gif, preview frames, and a disturbance summary."""
    trajectories = load_trajectories(source, second_source)
    frames = render_frames(
        trajectories,
        eye=eye,
        target=target,
        width=width,
        height=height,
        fovy=fovy,
        labels=labels,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mp4_path = output / "trajectory_traces.mp4"
    gif_path = output / "trajectory_traces.gif"
    _write_mp4(frames, mp4_path, fps)
    _write_gif(frames[::2], gif_path, fps / 2)
    preview_names = ("preview_start.png", "preview_mid.png", "preview_end.png")
    preview_ids = (0, len(frames) // 2, len(frames) - 1)
    previews = []
    for name, index in zip(preview_names, preview_ids):
        path = output / name
        _write_png(frames[index], path)
        previews.append(str(path))
    names = _labels(labels, len(trajectories))
    summary = {
        "source": str(source),
        "second_source": None if second_source is None else str(second_source),
        "trajectory_count": len(trajectories),
        "frame_count": len(frames),
        "fps": fps,
        "video": str(mp4_path),
        "gif": str(gif_path),
        "previews": previews,
        "episodes": [_episode_summary(episode, index, names[index]) for index, episode in enumerate(trajectories)],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def render_frames(
    trajectories,
    *,
    eye=SCENE1_CAMERA_EYE,
    target=SCENE1_CAMERA_TARGET,
    width=640,
    height=640,
    fovy=1.0,
    labels=None,
):
    """Build RGB frames. The first episode is the one whose objects move."""
    if not trajectories:
        raise ValueError("Need at least one trajectory")
    if len(trajectories) > len(TRACE_COLORS):
        raise ValueError("At most two trajectories can be drawn together")
    camera = Camera(eye, target, width, height, fovy)
    played = trajectories[0]
    steps = int(played["tcp_pose"].shape[0])
    names = _labels(labels, len(trajectories))
    hold = 12
    frames = []
    for _ in range(hold):
        frames.append(_draw(camera, trajectories, played, 0, names, planned=True))
    for step in range(steps):
        frames.append(_draw(camera, trajectories, played, step, names, planned=False))
    for _ in range(hold):
        frames.append(_draw(camera, trajectories, played, steps - 1, names, planned=False))
    return frames


class Camera:
    """Pinhole camera matching the scene's human render pose."""

    def __init__(self, eye, target, width, height, fovy):
        self.eye = np.asarray(eye, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64)
        forward = self.target - self.eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        self.axes = np.stack((right, up, forward), axis=0)
        self.focal = (height / 2.0) / np.tan(float(fovy) / 2.0)
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.width = int(width)
        self.height = int(height)

    def project(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        camera = (points - self.eye) @ self.axes.T
        depth = camera[:, 2]
        pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
        valid = depth > 0.02
        pixels[valid, 0] = self.focal * camera[valid, 0] / depth[valid] + self.cx
        pixels[valid, 1] = self.cy - self.focal * camera[valid, 1] / depth[valid]
        return pixels, depth


def _read_archive(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in ("tcp_pose", "distractor_poses", "object_sizes_m") if key not in archive.files]
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        data = {key: archive[key].copy() for key in archive.files}
    tcp = np.asarray(data["tcp_pose"])
    if not (tcp.ndim == 3 and tcp.shape[0] == 2):
        _validate_episode(data, path)
    return data


def _validate_episode(data, path):
    tcp = np.asarray(data["tcp_pose"])
    poses = np.asarray(data["distractor_poses"])
    sizes = np.asarray(data["object_sizes_m"])
    if tcp.ndim != 2 or tcp.shape[1] != 7:
        raise ValueError(f"{path} tcp_pose must be [T,7]")
    if poses.ndim != 3 or poses.shape[0] != tcp.shape[0] or poses.shape[2] != 7:
        raise ValueError(f"{path} distractor poses must be [T,N,7] aligned with tcp_pose")
    if sizes.shape != (poses.shape[1],) or np.any(sizes <= 0):
        raise ValueError(f"{path} object_sizes_m must be one positive length per distractor")
    if "target_pose" in data and np.asarray(data["target_pose"]).shape != tcp.shape:
        raise ValueError(f"{path} target_pose must match tcp_pose")


def _split_stacked(data):
    tcp = np.asarray(data["tcp_pose"])
    if tcp.ndim != 3 or tcp.shape[0] != 2:
        return None
    episodes = []
    for index in range(2):
        episode = {}
        for key, value in data.items():
            array = np.asarray(value)
            if array.shape[:1] == (2,):
                episode[key] = array[index]
            else:
                episode[key] = array
        _validate_episode(episode, f"stacked trajectory {index}")
        episodes.append(episode)
    return episodes


def _labels(labels, count):
    defaults = ("Refined trajectory", "Original trajectory") if count == 2 else ("TCP trace",)
    if labels is None:
        return defaults[:count]
    if len(labels) != count:
        raise ValueError("Label count must match the number of trajectories")
    return tuple(labels)


def _episode_summary(episode, index, label):
    score = disturbance_from_poses(
        episode["distractor_poses"],
        episode["object_sizes_m"],
        topple_flags=episode.get("distractor_toppled"),
        fall_flags=episode.get("distractor_fallen"),
    )
    return {
        "index": index,
        "label": label,
        "steps": int(episode["tcp_pose"].shape[0]),
        "dr_mean": score["dr_mean"],
        "dr_max": score["dr_max"],
        "topple_count": score.get("topple_count", 0),
        "fall_count": score.get("fall_count", 0),
        "per_object_dr": score["per_object_dr"],
    }


def _draw(camera, trajectories, played, step, names, planned):
    image = np.full((camera.height, camera.width, 3), 236, dtype=np.uint8)
    image[:, :] = (232, 228, 220)
    _draw_table(camera, image)
    _draw_goal(camera, image, played, step)
    _draw_boxes(camera, image, played, step, camera.eye)
    for index, episode in enumerate(trajectories):
        _draw_trace(camera, image, episode["tcp_pose"][:, :3], TRACE_COLORS[index])
    _draw_disturbance(camera, image, played, step)
    _draw_tcp_marker(camera, image, played["tcp_pose"][step])
    _draw_hud(image, trajectories, played, step, names, planned)
    return image


def _draw_table(camera, image):
    corners = np.array(
        [[-0.45, -0.40, 0.0], [0.40, -0.40, 0.0], [0.40, 0.40, 0.0], [-0.45, 0.40, 0.0]],
        dtype=np.float64,
    )
    _fill_polygon(camera, image, corners, (186, 154, 112))


def _draw_goal(camera, image, episode, step):
    if "goal_pos" not in episode:
        return
    goal = np.asarray(episode["goal_pos"], dtype=np.float64)
    center = goal[step] if goal.ndim == 2 else goal
    ring = []
    for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        ring.append([center[0] + 0.045 * np.cos(angle), center[1] + 0.045 * np.sin(angle), 0.002])
    pixels, _depth = camera.project(np.asarray(ring))
    if not np.isfinite(pixels).all():
        return
    import cv2

    cv2.polylines(image, [np.round(pixels).astype(np.int32)], True, (40, 170, 70), 2, cv2.LINE_AA)


def _draw_boxes(camera, image, episode, step, eye):
    faces = []
    faces.extend(_box_faces(
        episode["distractor_poses"][step],
        _box_dimensions(episode["object_sizes_m"]),
        _colors(episode["object_sizes_m"].shape[0]),
        eye,
    ))
    if "target_pose" in episode:
        faces.extend(_box_faces(
            episode["target_pose"][step][None, :],
            np.array([[0.04, 0.04, 0.04]]),
            np.array([[220, 50, 50]]),
            eye,
        ))
    faces.sort(key=lambda item: item[0], reverse=True)
    for _depth, corners, color in faces:
        _fill_polygon(camera, image, corners, color)


def _box_faces(poses, dimensions, colors, eye):
    faces = []
    eye = np.asarray(eye, dtype=np.float64)
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 7)
    light = np.array([0.3, 0.4, 0.85])
    for pose, size, color in zip(poses, dimensions, colors):
        rotation = _quat_to_matrix(pose[3:])
        corners = (_BOX_CORNERS * (np.asarray(size) / 2.0)) @ rotation.T + pose[:3]
        for face in _BOX_FACES:
            quad = corners[list(face)]
            normal = np.cross(quad[1] - quad[0], quad[2] - quad[0])
            norm = np.linalg.norm(normal)
            if norm < 1e-8:
                continue
            normal = normal / norm
            center = quad.mean(axis=0)
            if np.dot(normal, center - pose[:3]) < 0:
                normal = -normal
            shade = 0.55 + 0.45 * max(0.0, float(np.dot(normal, light)))
            depth = float(np.linalg.norm(center - eye))
            faces.append((depth, quad, np.clip(color * shade, 0, 255)))
    return faces


def _box_dimensions(sizes):
    sizes = np.asarray(sizes, dtype=np.float64)
    dims = np.repeat(sizes[:, None], 3, axis=1)
    tall = sizes > 0.05
    dims[tall, 0] = 0.035
    dims[tall, 1] = 0.035
    return dims


def _colors(count):
    if count <= len(DISTRACTOR_COLORS):
        return DISTRACTOR_COLORS[:count]
    repeats = np.resize(DISTRACTOR_COLORS, (count, 3))
    return repeats


def _draw_trace(camera, image, points, gradient):
    import cv2

    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return
    sample = points[:: max(1, len(points) // 28)]
    if not np.allclose(sample[-1], points[-1]):
        sample = np.concatenate((sample, points[-1:]), axis=0)
    pixels, depth = camera.project(sample)
    start, end = gradient
    for index in range(len(sample) - 1):
        if not (np.isfinite(pixels[index]).all() and np.isfinite(pixels[index + 1]).all()):
            continue
        color = _mix(start, end, index / max(1, len(sample) - 1))
        thickness = 2
        cv2.line(
            image,
            tuple(np.round(pixels[index]).astype(int)),
            tuple(np.round(pixels[index + 1]).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )
    for index, (pixel, dist) in enumerate(zip(pixels, depth)):
        if not np.isfinite(pixel).all():
            continue
        color = _mix(start, end, index / max(1, len(sample) - 1))
        radius = int(np.clip(7.0 * (0.8 / max(dist, 0.2)), 3, 8))
        center = tuple(np.round(pixel).astype(int))
        cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, center, radius, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_disturbance(camera, image, episode, step):
    import cv2

    poses = episode["distractor_poses"][: step + 1, :, :3]
    if poses.shape[0] < 2:
        return
    start = poses[0]
    current = poses[-1]
    for index in range(poses.shape[1]):
        moved = float(np.linalg.norm(current[index] - start[index]))
        if moved < 0.004:
            continue
        pixels, _depth = camera.project(poses[:, index, :])
        finite = np.isfinite(pixels).all(axis=1)
        if finite.sum() < 2:
            continue
        poly = np.round(pixels[finite]).astype(np.int32)
        heat = np.clip(moved / 0.08, 0.0, 1.0)
        color = _mix((255, 190, 120), (220, 60, 20), heat)
        cv2.polylines(image, [poly], False, color, 2, cv2.LINE_AA)
        flags = episode.get("distractor_toppled")
        if flags is not None and bool(flags[step, index]):
            cv2.circle(image, tuple(poly[-1]), 10, (220, 30, 30), 2, cv2.LINE_AA)


def _draw_tcp_marker(camera, image, pose):
    import cv2

    pixel, depth = camera.project(pose[:3])
    if not np.isfinite(pixel).all():
        return
    center = tuple(np.round(pixel[0]).astype(int))
    radius = int(np.clip(10.0 * (0.8 / max(float(depth[0]), 0.2)), 4, 12))
    cv2.circle(image, center, radius, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(image, center, radius, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_hud(image, trajectories, played, step, names, planned):
    import cv2

    panel = image.copy()
    cv2.rectangle(panel, (12, 12), (300, 118 + 28 * len(trajectories)), (255, 255, 255), -1)
    cv2.addWeighted(panel, 0.82, image, 0.18, 0, image)
    title = "Planned traces" if planned else f"Step {step + 1}/{played['tcp_pose'].shape[0]}"
    cv2.putText(image, title, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    y = 64
    for index, episode in enumerate(trajectories):
        color = TRACE_COLORS[index][1]
        cv2.circle(image, (32, y - 4), 6, color, -1, cv2.LINE_AA)
        cv2.putText(image, names[index], (48, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        y += 24
    if planned:
        score = _score_until(played, played["tcp_pose"].shape[0] - 1)
        caption = "Episode disturbance"
    else:
        score = _score_until(played, step)
        caption = "Disturbance so far"
    cv2.putText(image, caption, (24, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
    _draw_bar(image, 24, y + 16, 180, score["dr_mean"])
    text = f"{score['dr_mean']:.3f}   topple {score.get('topple_count', 0)}  fall {score.get('fall_count', 0)}"
    cv2.putText(image, text, (24, y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)


def _score_until(episode, step):
    stop = episode["tcp_pose"].shape[0] - 1 if step is None else step
    topple = episode.get("distractor_toppled")
    fall = episode.get("distractor_fallen")
    return disturbance_from_poses(
        episode["distractor_poses"],
        episode["object_sizes_m"],
        stop=stop,
        topple_flags=None if topple is None else topple,
        fall_flags=None if fall is None else fall,
    )


def _draw_bar(image, x, y, width, value):
    import cv2

    filled = int(np.clip(value / 1.0, 0.0, 1.0) * width)
    cv2.rectangle(image, (x, y), (x + width, y + 14), (255, 255, 255), 1, cv2.LINE_AA)
    if filled > 0:
        cv2.rectangle(image, (x, y), (x + filled, y + 14), (220, 90, 40), -1, cv2.LINE_AA)


def _fill_polygon(camera, image, corners, color):
    import cv2

    pixels, depth = camera.project(corners)
    if not np.isfinite(pixels).all() or np.any(depth <= 0.02):
        return
    cv2.fillConvexPoly(image, np.round(pixels).astype(np.int32), tuple(int(channel) for channel in color), cv2.LINE_AA)


def _quat_to_matrix(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-8:
        raise ValueError("Invalid zero quaternion")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _mix(start, end, fraction):
    mix = (1.0 - fraction) * np.asarray(start) + fraction * np.asarray(end)
    return tuple(int(channel) for channel in np.clip(mix, 0, 255))


def _write_png(frame, path):
    import cv2

    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def _write_mp4(frames, path, fps):
    import cv2

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open a video writer for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if path.stat().st_size < 1000:
        raise RuntimeError(f"Video writer produced an empty file at {path}")


def _write_gif(frames, path, fps):
    import imageio.v2 as imageio

    imageio.mimsave(path, frames, duration=1.0 / max(float(fps), 1.0), loop=0)
