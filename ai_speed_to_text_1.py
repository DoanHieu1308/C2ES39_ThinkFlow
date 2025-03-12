from dataclasses import dataclass
import json
from typing import Dict, List, Union
from datasets import Dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import torch
import evaluate
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from laydulieu import metadata

# Đọc dữ liệu từ file JSON
data_json = metadata()
data = json.loads(data_json)

# Phân chia dữ liệu thành tập train và test dựa trên trường "folder"
train_data = [item for item in data if item["folder"] == "train"]
test_data = [item for item in data if item["folder"] == "test"]

common_voice = DatasetDict()

# Tạo dataset cho tập train và test
common_voice["train"] = Dataset.from_dict({
    "audio": [item["audio_path"] for item in train_data],
    "sentence": [item["text"] for item in train_data]
})

common_voice["test"] = Dataset.from_dict({
    "audio": [item["audio_path"] for item in test_data],
    "sentence": [item["text"] for item in test_data]
})

# Prepare Feature Extractor, Tokenizer and Data
feature_extrator = WhisperFeatureExtractor.from_pretrained("openai/whisper-large")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large", language = "vietnamese", task = "transcribe")

# Combine elements with WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-large", language = "vietnamese", task = "transcribe")

print("| Check the random audio example from Common voice dataset to see what form the data is in: ")
print(f"{common_voice['train'][0]}\n")

# Downsample from 48kHz to 16kHz
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

print("| Check the effect of downsampling")
print(f"{common_voice['train'][0]}\n")

# Prepare and use function to prepare our data ready for the Whisper AI model 
def prepare_dataset(batch):
    # Prepare audio data to be suitable for Whisper AI model
    # (1) Load and resample input feature from input audio array 
    audio = batch["audio"]

    # (2) Compute log Mel input features from input audio array
    batch["input_features"] = feature_extrator(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # (3) Encode target text to lable ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch
    

# Prepare and use funcition to prepare our data ready for the Whisper AI model
common_voice = common_voice.map(
    prepare_dataset,
    remove_columns=common_voice.column_names["train"],
    num_proc=1
)


# Training and evaluation
# (1) Initialize the Data collator

@dataclass
class DataCollatorSeq2SeqWithPadding: 
    """
       Use Data Collator to perform Speech Seq2Seq with padding
    """
    processor : any

    def __call__(self, features: list[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]

        lables_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = lables_batch["input_ids"].masked_fill(lables_batch.attention_mask.ne(1), -100)

        if(labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels

        return batch


data_collator = DataCollatorSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")

# STEP 5.3 Load a pre-trained checkpoint
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")

"""
Overide generation arguments:
- no tokens are forced as decoder outputs: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.forced_decoder_ids
- no tokens are suppressed during generation: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.suppress_tokens
"""
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


# STEP 5.4 Define the training configration 
"""
Check for Seq2SeqTrainingArguments here:
https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments
"""

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-vn",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False, # testing
)

# Initialize a trainer.
"""
Forward the training arguments to the Hugging Face trainer along with our model,
dataset, data collator and compute_metrics function.
"""
def compute_metrics(pred):
    # Define evaluation metric. We will use Word Error Rate (WER) metric
    # For more information check: 
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens = True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens = True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer" : wer}

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# STEP 5.5 Training 
"""
   Training will take appr 5-10 hours depending on your GPU
"""

has_labels = any("labels" in sample for sample in common_voice["train"])
print(has_labels)


print("Training is started.")
trainer.train()
print("Training is finished.")

