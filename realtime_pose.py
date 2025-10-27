import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

KEY_IDX_13 = [
    0,   # nose
    11, 12,  # shoulders
    23, 24,  # hips
    25, 26,  # knees
    27, 28,  # ankles
    15, 16,  # wrists
    29, 30,  # heels
    31, 32   # feet index
]

# Các cặp điểm để vẽ xương (tùy chọn)
SKELETON_13 = [
    (11, 12), (11, 23), (12, 24),  # shoulders - hips
    (23, 24), (23, 25), (24, 26),  # hips - knees
    (25, 27), (26, 28),            # knees - ankles
    (11, 15), (12, 16),            # shoulders - wrists
    (27, 29), (28, 30), (29, 31), (30, 32)  # feet
]

def normalize_landmarks(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    pts_13 = pts[KEY_IDX_13]
    center = np.mean(pts_13, axis=0)
    pts_13 -= center
    scale = np.linalg.norm(pts[11] - pts[12]) if np.linalg.norm(pts[11] - pts[12]) > 1e-6 else 1.0
    pts_13 /= scale
    return pts_13

# === Realtime webcam ===
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            pts = np.array([[lm.x * w, lm.y * h] for lm in results.pose_landmarks.landmark])
            pts_13 = pts[KEY_IDX_13]

            # --- Vẽ 13 keypoints ---
            for x, y in pts_13:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

            # --- Vẽ xương (tùy chọn) ---
            for (i, j) in SKELETON_13:
                if i < len(results.pose_landmarks.landmark) and j < len(results.pose_landmarks.landmark):
                    p1 = (int(results.pose_landmarks.landmark[i].x * w),
                          int(results.pose_landmarks.landmark[i].y * h))
                    p2 = (int(results.pose_landmarks.landmark[j].x * w),
                          int(results.pose_landmarks.landmark[j].y * h))
                    cv2.line(frame, p1, p2, (255, 0, 0), 2)

            # Chuẩn hóa 13 điểm (cho inference)
            pts_13_norm = normalize_landmarks(results.pose_landmarks)
            print(np.round(pts_13_norm, 3))

        cv2.imshow("BlazePose - 13 Keypoints Only", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
