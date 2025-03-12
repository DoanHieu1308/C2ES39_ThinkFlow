import json
import os

# Đường dẫn tới thư mục chứa dataset
BASE_PATH = "E:\\CAPTONE2\\processed_dataset1"

# Hàm để xử lý metadata từ thư mục con (train hoặc test)
def process_metadata(folder_name):
    dataset = []
    metadata_path = os.path.join(BASE_PATH, folder_name, "metadata.json")

    # Đọc file metadata.json
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Đường dẫn tới thư mục chứa file audio
    audio_dir = os.path.join('processed_dataset1', folder_name)

    # Xử lý từng mục trong metadata
    for item in metadata:
        file_name = item['file_name']
        text = item['text']

        # Tạo đường dẫn đến file audio
        audio_path = os.path.join(audio_dir, file_name).replace("\\", "/")

        # Kiểm tra sự tồn tại của file audio
        if os.path.exists(os.path.join(BASE_PATH, folder_name, file_name)):
            dataset.append({
                "folder" : folder_name,
                "audio_path": "E:/CAPTONE2/"+audio_path,
                "text": text
            })
        else:
            print(f"File {file_name} không tồn tại trong {folder_name}.")

    return dataset

def metadata():
    # Lấy dữ liệu từ cả train và test
    train_data = process_metadata("train")
    test_data = process_metadata("test")

    # Gộp dữ liệu từ train và test
    full_dataset = train_data + test_data
    full_dataset_last = json.dumps(full_dataset, ensure_ascii=False, indent=4)

    return full_dataset_last

# print(metadata())