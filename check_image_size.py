import numpy as np
data = np.load('fall_01_cam0_lstm.flattened.npy')
print(data.shape)
print("Min:", np.min(data))
print("Max:", np.max(data))
print("Mean:", np.mean(data))

import cv2

img_path = "UR Fall/fall/fall-01-cam0-rgb/fall-01-cam0-rgb-001.png"

img = cv2.imread(img_path)
if img is None:
    print(f"Không đọc được ảnh: {img_path}")
else:
    h, w, c = img.shape
    print(f"Ảnh: {img_path}")
    print(f"  Chiều rộng (width): {w}")
    print(f"  Chiều cao (height): {h}")
    print(f"  Số kênh màu: {c}")
import os
import cv2

# ===================== CONFIG =====================
VIDEO_ROOT = "Mydata_mp4"   # Thư mục chứa video cần kiểm tra
EXTS = (".mp4", ".mov")     # Đuôi video cần quét

# ===================== HÀM LẤY FPS =====================
def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video: {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

# ===================== DUYỆT TOÀN BỘ =====================
for root, dirs, files in os.walk(VIDEO_ROOT):
    for f in files:
        if f.lower().endswith(EXTS):
            path = os.path.join(root, f)
            fps = get_fps(path)
            if fps is not None:
                print(f"{path}: {fps:.2f} FPS")

