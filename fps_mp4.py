import os
import subprocess

INPUT_ROOT = "Mydata"
OUTPUT_ROOT = "Mydata_mp4"
TARGET_FPS = 30

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def convert_with_ffmpeg(input_path, output_path, target_fps):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", f"fps={target_fps},format=yuv420p",  # Áp dụng FPS & chuẩn hóa định dạng
        "-c:v", "libx264",                          # Mã hóa H.264 chuẩn
        "-c:a", "copy",                             # Giữ nguyên âm thanh
        "-metadata:s:v", "rotate=0",                # Xóa tag xoay (đã áp dụng)
        "-y", output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✅ {input_path} -> {output_path}")

for label in ["fall", "non_fall"]:
    in_dir = os.path.join(INPUT_ROOT, label)
    out_dir = os.path.join(OUTPUT_ROOT, label)
    os.makedirs(out_dir, exist_ok=True)

    for file in os.listdir(in_dir):
        if file.lower().endswith((".mov", ".mp4")):
            in_path = os.path.join(in_dir, file)
            out_path = os.path.join(out_dir, os.path.splitext(file)[0] + "_30fps.mp4")
            convert_with_ffmpeg(in_path, out_path, TARGET_FPS)

print("🎯 Hoàn tất chuyển đổi MOV → MP4, 30 FPS, đúng hướng hiển thị!")
