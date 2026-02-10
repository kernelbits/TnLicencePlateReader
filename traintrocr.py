import os
import pandas as pd
from datasets import Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from PIL import Image
import evaluate

# --- Config ---
DATASET_PATH = "./data"
MODEL_NAME = "microsoft/trocr-large-printed"
OUTPUT_DIR = "./trocr_model"
BATCH_SIZE = 1
EPOCHS = 10
MAX_TARGET_LENGTH = 16

# --- Load dataset ---
df = pd.read_csv(os.path.join(DATASET_PATH, "labels.csv"))

def load_image(example):
    image_path = os.path.join(DATASET_PATH, "images", example["filename"])
    example["image"] = Image.open(image_path).convert("RGB")
    return example

dataset = Dataset.from_pandas(df)
dataset = dataset.map(load_image)

# --- Train/Val split ---
dataset = dataset.train_test_split(test_size=0.1)
train_ds = dataset["train"]
val_ds = dataset["test"]

# --- Processor & Model ---
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# 🔥 VRAM saver
model.gradient_checkpointing_enable()

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- Preprocessing ---
def preprocess(example):
    pixel_values = processor(images=example["image"], return_tensors="pt").pixel_values
    labels = processor.tokenizer(
        example["text"],
        padding="max_length",
        max_length=MAX_TARGET_LENGTH,
        truncation=True
    ).input_ids

    example["pixel_values"] = pixel_values.squeeze()
    example["labels"] = labels
    return example

train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

# --- Data collator ---
def collate_fn(batch):
    pixel_values = torch.stack([torch.tensor(x["pixel_values"]) for x in batch])
    labels = torch.stack([torch.tensor(x["labels"]) for x in batch])
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }

# --- Metric ---
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

# --- Training args (CRASH SAFE) ---
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    num_train_epochs=EPOCHS,
    logging_steps=20,

    # 🔴 SAVE OFTEN
    save_steps=500,
    save_total_limit=3,

    evaluation_strategy="epoch",
    save_strategy="steps",

    predict_with_generate=True,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics,
)

# 🔥 AUTO RESUME
checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    if len(checkpoints) > 0:
        checkpoint = os.path.join(OUTPUT_DIR, sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1])
        print("Resuming from:", checkpoint)

trainer.train(resume_from_checkpoint=checkpoint)

trainer.save_model(OUTPUT_DIR)
print("Training done. Model saved in", OUTPUT_DIR)
