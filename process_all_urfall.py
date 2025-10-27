import os
import cv2
import numpy as np
import mediapipe as mp
from glob import glob
from pathlib import Path
import re
from typing import List

# ===================== USER CONFIG =====================
DATA_ROOT = "UR Fall"               # Thư mục gốc chứa fall / non_fall
OUTPUT_ROOT = "UR_Fall_normalized"  # Thư mục lưu kết quả
FPS = 30.0
CONF_THRESH = 0.1
KEYPOINT_ORDER = [
    'Nose',
    'Left Shoulder', 'Right Shoulder',
    'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist',
    'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee',
    'Left Ankle', 'Right Ankle'
]
# =======================================================

mp_pose = mp.solutions.pose
LANDMARK_MAP = {name: getattr(mp_pose.PoseLandmark, name.upper().replace(" ", "_")) for name in KEYPOINT_ORDER}


def natural_sort_key(s: str):
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_images(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg")
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(folder, ext)))
    return sorted(files, key=lambda p: natural_sort_key(os.path.basename(p)))


def extract_keypoints_from_image(img_bgr):
    img_h, img_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True,
                      model_complexity=2,
                      enable_segmentation=False,
                      min_detection_confidence=0.5) as pose:
        res = pose.process(img_rgb)
    if not res.pose_landmarks:
        return None
    landmarks = res.pose_landmarks.landmark
    kp = []
    for name in KEYPOINT_ORDER:
        lm = landmarks[LANDMARK_MAP[name].value]
        x = lm.x * img_w
        y = lm.y * img_h
        c = getattr(lm, "visibility", getattr(lm, "presence", 0.0))
        kp.append((x, y, float(c)))
    return np.array(kp, dtype=float)


def compute_centroid_and_scale(kps: np.ndarray, conf_thresh=CONF_THRESH, prev_centroid=None, img_height=480):
    MIN_SCALE = img_height / 10
    coords = kps[:, :2]
    conf = kps[:, 2]
    left_sh, right_sh = coords[1], coords[2]
    left_hip, right_hip = coords[7], coords[8]

    sh_points, hip_points = [], []
    if conf[1] > conf_thresh: sh_points.append(left_sh)
    if conf[2] > conf_thresh: sh_points.append(right_sh)
    if conf[7] > conf_thresh: hip_points.append(left_hip)
    if conf[8] > conf_thresh: hip_points.append(right_hip)

    mid_sh = np.array(sh_points).mean(axis=0) if sh_points else None
    mid_hip = np.array(hip_points).mean(axis=0) if hip_points else None

    if mid_sh is not None and mid_hip is not None:
        centroid = (mid_sh + mid_hip) / 2.0
        scale = max(np.linalg.norm(mid_sh - mid_hip), MIN_SCALE)
    elif mid_sh is not None:
        centroid, scale = mid_sh, MIN_SCALE
    elif mid_hip is not None:
        centroid, scale = mid_hip, MIN_SCALE
    else:
        centroid = prev_centroid.copy() if prev_centroid is not None else coords.mean(axis=0)
        scale = MIN_SCALE
    return centroid, scale


def normalize_keypoints_frame(kps: np.ndarray, centroid, scale):
    coords = (kps[:, :2] - centroid[None, :]) / float(scale)
    conf = kps[:, 2].copy()
    return np.concatenate([coords, conf[:, None]], axis=1)


def interpolate_missing(series: np.ndarray):
    T = series.shape[0]
    for j in range(series.shape[1]):
        for c in range(2):
            vals = series[:, j, c]
            confs = series[:, j, 2]
            mask = confs > CONF_THRESH
            if mask.sum() == 0:
                continue
            indices = np.arange(T)
            vals[~mask] = np.interp(indices[~mask], indices[mask], vals[mask])
            series[:, j, c] = vals
    return series


def process_folder(images_dir: str, output_flattened: str, fps: float = 30.0):
    files = list_images(images_dir)
    if not files:
        print(f"[WARN] No images in {images_dir}")
        return

    print(f"[INFO] Processing {len(files)} frames from: {images_dir}")
    frames_kps = []

    for f in files:
        img = cv2.imread(f)
        kp = extract_keypoints_from_image(img) if img is not None else None
        if kp is None:
            kp = np.zeros((len(KEYPOINT_ORDER), 3), dtype=float)
        frames_kps.append(kp)

    frames_kps = np.stack(frames_kps, axis=0)
    frames_kps = interpolate_missing(frames_kps)

    normalized = np.zeros_like(frames_kps)
    prev_centroid = None
    prev_angle = 0.0  # nếu frame đầu tiên thiếu góc thì mặc định góc = 0
    angles = []

    for t in range(frames_kps.shape[0]):
        centroid, scale = compute_centroid_and_scale(frames_kps[t], prev_centroid=prev_centroid)
        normalized[t] = normalize_keypoints_frame(frames_kps[t], centroid, scale)
        prev_centroid = centroid

        # --- Tính góc thân người ---
        coords = frames_kps[t][:, :2]
        conf = frames_kps[t][:, 2]
        left_sh, right_sh = coords[1], coords[2]
        left_hip, right_hip = coords[7], coords[8]

        sh_points, hip_points = [], []
        if conf[1] > CONF_THRESH: sh_points.append(left_sh)
        if conf[2] > CONF_THRESH: sh_points.append(right_sh)
        if conf[7] > CONF_THRESH: hip_points.append(left_hip)
        if conf[8] > CONF_THRESH: hip_points.append(right_hip)

        mid_sh = np.array(sh_points).mean(axis=0) if len(sh_points) > 0 else None
        mid_hip = np.array(hip_points).mean(axis=0) if len(hip_points) > 0 else None

        if mid_sh is not None and mid_hip is not None:
            vec_body = mid_hip - mid_sh
            vec_vert = np.array([0, -1])  # hướng lên trên
            norm_b = np.linalg.norm(vec_body)
            if norm_b < 1e-6:
                angle = prev_angle
            else:
                cos_theta = np.clip(np.dot(vec_body, vec_vert) / norm_b, -1.0, 1.0)
                angle = np.arccos(cos_theta)  # radians
        else:
            angle = prev_angle  # fallback nếu thiếu cả vai/hông

        prev_angle = angle
        angles.append(angle)

    angles = np.array(angles)[:, None]  # (T, 1)

    # --- Tính vận tốc keypoint ---
    dt = 1.0 / fps
    velocities = np.zeros((frames_kps.shape[0], len(KEYPOINT_ORDER)))
    for t in range(1, frames_kps.shape[0]):
        delta = normalized[t, :, :2] - normalized[t - 1, :, :2]
        velocities[t] = np.linalg.norm(delta, axis=1) / dt

    # Flatten -> (T, 13*4 + 1)
    out = np.zeros((frames_kps.shape[0], len(KEYPOINT_ORDER), 4))
    out[:, :, :3] = normalized
    out[:, :, 3] = velocities
    flattened = out.reshape(out.shape[0], -1)
    flattened = np.concatenate([flattened, angles], axis=1)

    os.makedirs(os.path.dirname(output_flattened), exist_ok=True)
    np.save(output_flattened, flattened)
    print(f"✅ Saved flattened: {output_flattened} | shape={flattened.shape}")


# ---------------- BATCH PROCESS ----------------
if __name__ == "__main__":
    for category in ["fall", "non_fall"]:
        input_dir = os.path.join(DATA_ROOT, category)
        output_dir = os.path.join(OUTPUT_ROOT, category)
        os.makedirs(output_dir, exist_ok=True)

        subfolders = sorted([d for d in glob(os.path.join(input_dir, "*")) if os.path.isdir(d)])
        print(f"\n=== Category: {category} ({len(subfolders)} folders) ===")

        for sub in subfolders:
            name = os.path.basename(sub)
            output_path = os.path.join(output_dir, f"{name}.flattened.npy")
            process_folder(sub, output_path, FPS)
