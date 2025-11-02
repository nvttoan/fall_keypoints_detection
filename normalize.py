import os
import cv2
import numpy as np
import mediapipe as mp
from glob import glob
import re
from pathlib import Path
from typing import List

IMAGES_DIR = "UR Fall/fall/fall-01-cam0-rgb"
OUTPUT_NPY = "fall_01_cam0_lstm.npy"
FPS = 30.0
CONF_THRESH = 0.2
KEYPOINT_ORDER = [
    'Nose',
    'Left Shoulder', 'Right Shoulder',
    'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist',
    'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee',
    'Left Ankle', 'Right Ankle'
]

# truy xuất landmark dùng LANDMARK_MAP[name].value.
mp_pose = mp.solutions.pose
LANDMARK_MAP = {name: getattr(mp_pose.PoseLandmark, name.upper().replace(" ", "_")) for name in KEYPOINT_ORDER}

# sap xep frame theo tên file
def natural_sort_key(s: str):
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]
#' tra ve danh sach anh trong thu muc
def list_images(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg")
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(folder, ext)))
    return sorted(files, key=lambda p: natural_sort_key(os.path.basename(p)))
# lay keypoints tu anh
def extract_keypoints_from_image(img_bgr):
    img_h, img_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) #BGR → RGB
    # tao object pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=2,
                      enable_segmentation=False, min_detection_confidence=0.5) as pose:
        res = pose.process(img_rgb)
    if not res.pose_landmarks:
        return None
    landmarks = res.pose_landmarks.landmark
    kp = []
    for name in KEYPOINT_ORDER: # lay 13 keypoints 
        lm = landmarks[LANDMARK_MAP[name].value]
        x = lm.x * img_w # chuyen toan bo ve toa do pixel
        y = lm.y * img_h
        c = getattr(lm, "visibility", getattr(lm, "presence", 0.0))
        kp.append((x, y, float(c)))
    return np.array(kp, dtype=float)

def compute_centroid_and_scale(kps: np.ndarray, conf_thresh=CONF_THRESH, prev_centroid=None, img_height=480):
    MIN_SCALE = img_height / 10 # gia tri scale nho nhat de tranh chia 0
    coords, conf = kps[:, :2], kps[:, 2]
    left_sh, right_sh = coords[1], coords[2]
    left_hip, right_hip = coords[7], coords[8]

    sh_points, hip_points = [], [] # lay diem vai va hong co do tin cay
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
    coords = (kps[:, :2] - centroid[None, :]) / float(scale) # chuan hoa scale
    conf = kps[:, 2].copy()
    print(f"    Norm check -> min: {coords.min():.4f}, max: {coords.max():.4f}, mean: {coords.mean():.4f}")
    return np.concatenate([coords, conf[:, None]], axis=1)

# noi suy gia tri thieu tu cac frame tin cay 
def interpolate_missing(series: np.ndarray):
    T = series.shape[0] # so frame
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
        return None

    print(f"[INFO] Processing {len(files)} frames from: {images_dir}")
    frames_kps = []

    for f in files:
        img = cv2.imread(f)
        kp = extract_keypoints_from_image(img) if img is not None else None
        if kp is None: # khong lay duoc keypoints thi gan toan bo 0( de co thoi gian)
            kp = np.zeros((len(KEYPOINT_ORDER), 3), dtype=float)
        frames_kps.append(kp)

    frames_kps = np.stack(frames_kps, axis=0)
    frames_kps = interpolate_missing(frames_kps)

    normalized = np.zeros_like(frames_kps)
    prev_centroid = None
    prev_angle = 0.0
    angles = []

    for t in range(frames_kps.shape[0]):
        centroid, scale = compute_centroid_and_scale(frames_kps[t], prev_centroid=prev_centroid)
        normalized[t] = normalize_keypoints_frame(frames_kps[t], centroid, scale)
        prev_centroid = centroid

        # --- góc thân người ---
        coords = frames_kps[t][:, :2]
        conf = frames_kps[t][:, 2]
        sh_points, hip_points = [], []
        if conf[1] > CONF_THRESH: sh_points.append(coords[1])
        if conf[2] > CONF_THRESH: sh_points.append(coords[2])
        if conf[7] > CONF_THRESH: hip_points.append(coords[7])
        if conf[8] > CONF_THRESH: hip_points.append(coords[8])

        mid_sh = np.array(sh_points).mean(axis=0) if sh_points else None
        mid_hip = np.array(hip_points).mean(axis=0) if hip_points else None

        if mid_sh is not None and mid_hip is not None:
            vec_body = mid_hip - mid_sh
            vec_vert = np.array([0, -1])
            norm_b = np.linalg.norm(vec_body)
            if norm_b < 1e-6:
                angle = prev_angle
            else:
                cos_theta = np.clip(np.dot(vec_body, vec_vert) / norm_b, -1.0, 1.0)
                angle = np.arccos(cos_theta)
        else:
            angle = prev_angle

        prev_angle = angle
        angles.append(angle)

    angles = np.array(angles)[:, None]

    dt = 1.0 / fps
    velocities = np.zeros((frames_kps.shape[0], len(KEYPOINT_ORDER)))
    for t in range(1, frames_kps.shape[0]):
        delta = normalized[t, :, :2] - normalized[t - 1, :, :2]
        velocities[t] = np.linalg.norm(delta, axis=1) / dt

    out = np.zeros((frames_kps.shape[0], len(KEYPOINT_ORDER), 4))
    out[:, :, :3] = normalized
    out[:, :, 3] = velocities
    flattened = out.reshape(out.shape[0], -1)
    flattened = np.concatenate([flattened, angles], axis=1)

    os.makedirs(os.path.dirname(output_flattened) or ".", exist_ok=True)
    np.save(output_flattened, flattened)
    print(f"Saved flattened: {output_flattened} | shape={flattened.shape}")
    return flattened

if __name__ == "__main__":
    flattened = process_folder(IMAGES_DIR, OUTPUT_NPY, FPS)
    if flattened is not None:
        np.save(Path(OUTPUT_NPY).with_suffix('.flattened.npy'), flattened)
        print(f"Also saved flattened version: {Path(OUTPUT_NPY).with_suffix('.flattened.npy')}")
