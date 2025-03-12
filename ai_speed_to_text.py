from datasets import load_from_disk, DatasetDict
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import TrainingArguments, Trainer
import torch
import librosa

# ✅ Load dataset đã xử lý
dataset = load_from_disk("E:\\CAPTONE2\\processed_dataset")

if len(dataset) > 1:
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
else:
    print("Không đủ dữ liệu để chia train/test! Dùng toàn bộ làm train.")
    train_dataset = dataset
    eval_dataset = None

# ✅ Load processor & model của Whisper
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")

def prepare_data(batch):
    audio = batch["audio"]["array"]
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    labels = processor.tokenizer(batch["text"], return_tensors="pt").input_ids[0]
    return {"input_values": inputs.input_features[0], "labels": labels}

dataset = dataset.map(prepare_data, remove_columns=["audio", "text"])

# ✅ Cấu hình tham số huấn luyện
training_args = TrainingArguments(
    output_dir="./whisper_finetuned",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    eval_strategy="epoch" if eval_dataset else "no",  # ✅ Fix lỗi thiếu eval_dataset
    save_strategy="epoch",
    learning_rate=1e-4,
    warmup_steps=500,
    logging_dir="./logs",
    num_train_epochs=5,
    save_total_limit=2,
    fp16=True,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset if eval_dataset else None,  # ✅ Fix lỗi thiếu eval_dataset
    tokenizer=processor  # ✅ Đúng tham số, không phải "processing_class"
)

# ✅ Bắt đầu huấn luyện
trainer.train()

# ✅ Lưu model sau khi fine-tune
model.save_pretrained("whisper_vietnamese")
processor.save_pretrained("whisper_vietnamese")

# ✅ Load model đã fine-tune
processor = WhisperProcessor.from_pretrained("whisper_vietnamese")
model = WhisperForConditionalGeneration.from_pretrained("whisper_vietnamese")

# ✅ Load file test & xử lý âm thanh
audio_file = "E:\\CAPTONE2\\Audio_train\\audio_giong_hue_1.mp3"
audio_array, _ = librosa.load(audio_file, sr=16000)

inputs = processor(audio_array, return_tensors="pt", sampling_rate=16000)

# ✅ Dự đoán transcript
with torch.no_grad():
    predicted_ids = model.generate(inputs.input_features)

# ✅ Chuyển kết quả thành text
transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print("Transcript:", transcript[0])
