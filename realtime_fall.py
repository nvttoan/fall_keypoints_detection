import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque

SEQ_LEN = 30
FPS = 30.0
CONF_THRESH = 0.1
DEVICE = torch.device("cpu")

MODEL_PATH = "model_final_1.pt"  

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

KEYPOINT_ORDER = [
    'Nose',
    'Left Shoulder', 'Right Shoulder',
    'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist',
    'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee',
    'Left Ankle', 'Right Ankle'
]
LANDMARK_MAP = {name: getattr(mp_pose.PoseLandmark, name.upper().replace(" ", "_")) for name in KEYPOINT_ORDER}

# LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size=128, num_layers=2, dropout=0.3, bidir=False):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_size, num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0,
                           bidirectional=bidir)
        r_out = hidden_size * (2 if bidir else 1)
        self.fc = nn.Sequential(
            nn.Linear(r_out, max(r_out // 2, 16)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(r_out // 2, 16), 1)
        )
        self.bidir = bidir
        self.num_layers = num_layers

    def forward(self, x, lengths=None):
        _, (hn, _) = self.rnn(x)
        if self.bidir:
            if self.num_layers > 1:
                last_h = torch.cat([hn[-2], hn[-1]], dim=1)
            else:
                last_h = torch.cat([hn[0], hn[1]], dim=1)
        else:
            last_h = hn[-1]
        logits = self.fc(last_h).squeeze(1)
        return logits

INPUT_DIM = 13 * 4 + 1  # x,y,conf,velocity + angle
model = LSTMClassifier(input_dim=INPUT_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

def extract_keypoints(landmarks, img_w, img_h):
    kps = []
    for name in KEYPOINT_ORDER:
        lm = landmarks.landmark[LANDMARK_MAP[name].value]

        x, y = lm.x * img_w, lm.y * img_h
        c = getattr(lm, "visibility", getattr(lm, "presence", 0.0))
        kps.append((x, y, float(c)))
    return np.array(kps, dtype=float)

def compute_centroid_and_scale(kps, prev_centroid=None, img_height=480):
    coords = kps[:, :2]
    conf = kps[:, 2]
    left_sh, right_sh = coords[1], coords[2]
    left_hip, right_hip = coords[7], coords[8]

    sh_points, hip_points = [], []
    if conf[1] > CONF_THRESH: sh_points.append(left_sh)
    if conf[2] > CONF_THRESH: sh_points.append(right_sh)
    if conf[7] > CONF_THRESH: hip_points.append(left_hip)
    if conf[8] > CONF_THRESH: hip_points.append(right_hip)

    MIN_SCALE = img_height / 10
    mid_sh = np.array(sh_points).mean(axis=0) if len(sh_points) > 0 else None
    mid_hip = np.array(hip_points).mean(axis=0) if len(hip_points) > 0 else None

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

def normalize_keypoints(kps, centroid, scale):
    coords = (kps[:, :2] - centroid[None, :]) / float(scale)
    conf = kps[:, 2].copy()
    return np.concatenate([coords, conf[:, None]], axis=1)

def compute_body_angle(kps, prev_angle):
    coords = kps[:, :2]
    conf = kps[:, 2]
    left_sh, right_sh = coords[1], coords[2]
    left_hip, right_hip = coords[7], coords[8]

    sh_points, hip_points = [], []
    if conf[1] > CONF_THRESH: sh_points.append(left_sh)
    if conf[2] > CONF_THRESH: sh_points.append(right_sh)
    if conf[7] > CONF_THRESH: hip_points.append(left_hip)
    if conf[8] > CONF_THRESH: hip_points.append(right_hip)

    if len(sh_points) and len(hip_points):
        mid_sh = np.array(sh_points).mean(axis=0)
        mid_hip = np.array(hip_points).mean(axis=0)
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
    return angle

# Realtime 
cap = cv2.VideoCapture(0)
prev_centroid, prev_kps, prev_angle = None, None, 0.0
seq_buffer = deque(maxlen=SEQ_LEN)

with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                  enable_segmentation=False, min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            #  Vẽ  13 điểm 
            kps = extract_keypoints(results.pose_landmarks, w, h)
            for i, (x, y, c) in enumerate(kps):
                if c > CONF_THRESH:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
                    cv2.putText(frame, str(i + 1), (int(x) + 5, int(y) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Tính toán chuẩn hóa 
            centroid, scale = compute_centroid_and_scale(kps, prev_centroid, h)
            norm_kps = normalize_keypoints(kps, centroid, scale)
            angle = compute_body_angle(kps, prev_angle)

            # Vận tốc
            if prev_kps is not None:
                delta = norm_kps[:, :2] - prev_kps[:, :2]
                velocity = np.linalg.norm(delta, axis=1) / (1.0 / FPS)
            else:
                velocity = np.zeros(len(KEYPOINT_ORDER))
            prev_kps = norm_kps.copy()
            prev_centroid = centroid
            prev_angle = angle

            # Feature vector (13×4 + 1)
            out = np.zeros((len(KEYPOINT_ORDER), 4))
            out[:, :3] = norm_kps
            out[:, 3] = velocity
            feature_vec = np.concatenate([out.flatten(), [angle]])
            seq_buffer.append(feature_vec)

            # Dự đoán khi đủ chuỗi
            if len(seq_buffer) == SEQ_LEN:
                x = torch.tensor([seq_buffer], dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    logits = model(x)
                    prob = torch.sigmoid(logits).item()
                    pred = 1 if prob >= 0.5 else 0
                label = "FALL" if pred == 1 else "Non-Fall"
                color = (0, 0, 255) if pred == 1 else (0, 255, 0)
                if pred == 1:
                    print(f"[ALERT] ⚠️ FALL detected! Probability = {prob:.2f}")
                cv2.putText(frame, f"{label} ({prob:.2f})", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        else:
            cv2.putText(frame, "No person detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)


        cv2.imshow("Realtime Fall Detection", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
