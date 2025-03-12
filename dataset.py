import json
import os
from datasets import load_dataset, Dataset, Audio

# Đọc JSON chứa danh sách file audio và transcript
json_file = "E:\\CAPTONE2\\TRAIN_AI\\metadata.json"

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Chuyển đổi dữ liệu để load vào Hugging Face dataset
dataset_entries = []
for entry in data:
    audio_path = entry["audio"]
    text_path = entry["text"]

    # Đọc nội dung file text chứa transcript
    if os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            transcript = f.read().strip()
    else:
        transcript = ""

    dataset_entries.append({"audio": audio_path, "text": transcript})

# Tạo dataset từ danh sách entries
dataset = Dataset.from_list(dataset_entries)

# Chuyển cột "audio" thành dạng số hóa có thể xử lý
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Kiểm tra dữ liệu mẫu
print(dataset[0])

# Lưu dataset dưới dạng chuẩn để huấn luyện AI
dataset.save_to_disk("E:\\CAPTONE2\\processed_dataset")

print("✅ Dữ liệu đã sẵn sàng để train AI!")
