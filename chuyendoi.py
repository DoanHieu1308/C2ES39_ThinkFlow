import os
import json
import shutil
import random

# Định nghĩa đường dẫn thư mục
json_file = "E:\\CAPTONE2\\TRAIN_AI\\metadata.json"
output_dir = "E:\\CAPTONE2\\processed_dataset1"

train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

# Tạo thư mục train và test nếu chưa có
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Đọc file JSON
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Xáo trộn danh sách để đảm bảo phân bố ngẫu nhiên
random.shuffle(data)

# Chia dữ liệu theo tỷ lệ mong muốn
train_ratio = 0.8  # 🔥 Thay đổi giá trị này để điều chỉnh tỷ lệ train/test
split_index = int(train_ratio * len(data))

# Danh sách metadata
train_metadata = []
test_metadata = []

# Xử lý từng file trong danh sách
for idx, entry in enumerate(data):
    audio_path = entry["audio"]
    text_path = entry["text"]

    # Đổi tên file audio
    new_audio_name = f"audio_{idx + 1}.wav"

    # Xác định thư mục đích (train hoặc test)
    if idx < split_index:
        dest_folder = train_dir
        metadata_list = train_metadata
    else:
        dest_folder = test_dir
        metadata_list = test_metadata

    # Đường dẫn mới của file audio
    new_audio_path = os.path.join(dest_folder, new_audio_name)

    # Chuyển file audio vào thư mục train/test
    shutil.copy(audio_path, new_audio_path)

    # Đọc transcript
    transcript = ""
    if os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            transcript = f.read().strip()

    # Thêm thông tin vào metadata
    metadata_list.append({"file_name": new_audio_name, "text": transcript})

# Lưu metadata.json vào thư mục train và test
with open(os.path.join(train_dir, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(train_metadata, f, ensure_ascii=False, indent=4)

with open(os.path.join(test_dir, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(test_metadata, f, ensure_ascii=False, indent=4)

print(f"✅ Dữ liệu đã được chia theo tỷ lệ {train_ratio * 100}% train, {(1 - train_ratio) * 100}% test!")
