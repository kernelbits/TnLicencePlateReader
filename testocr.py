from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    output_dir="./debug",
    per_device_train_batch_size=2,
    evaluation_strategy="steps",
    save_steps=500,
    save_total_limit=2
)

print("Seq2SeqTrainingArguments works fine!")
print(args)
