"""
train.py — Fine-tune BanglaBERT for 7-class Bangla emotion classification.

Usage:
    python train.py                          # synthetic demo data
    python train.py --data dataset.csv       # your own CSV (columns: text, label)
    python train.py --data data.csv --epochs 10 --output my_model
"""

import os, json, argparse
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)

BASE_MODEL = "csebuetnlp/banglabert"

EMOTIONS   = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
EMOTIONS_BN = {
    "joy": "আনন্দ", "sadness": "দুঃখ", "anger": "রাগ",
    "fear": "ভয়", "surprise": "বিস্ময়", "disgust": "ঘৃণা", "neutral": "নিরপেক্ষ",
}
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL  = {i: e for i, e in enumerate(EMOTIONS)}

# Built-in synthetic samples — replace with real data for production
SAMPLES = [
    ("আজকে খুব ভালো লাগছে, সারাদিন মন আনন্দে ভরে আছে!", "joy"),
    ("বন্ধুদের সাথে আড্ডায় মনটা অনেক হালকা হয়ে গেল।", "joy"),
    ("পরীক্ষায় প্রথম হয়েছি, এত খুশি আর কখনো হইনি!", "joy"),
    ("নতুন চাকরি পেয়েছি, স্বপ্ন পূরণ হল অবশেষে।", "joy"),
    ("আজকে পরিবারের সবাইকে একসাথে দেখে মনটা ভরে গেল।", "joy"),
    ("মনটা আজকে খুব ভারী, কিছুতেই ভালো লাগছে না।", "sadness"),
    ("প্রিয়জন হারানোর বেদনা কখনো যায় না।", "sadness"),
    ("একা একা বসে আছি, চোখ থেকে পানি পড়ছে।", "sadness"),
    ("এত কষ্ট করেও কোনো ফল পাচ্ছি না।", "sadness"),
    ("পুরনো স্মৃতিগুলো মনে পড়লে বুকটা ভেঙে যায়।", "sadness"),
    ("এত অন্যায় সহ্য করা যাচ্ছে না, রাগে মাথা গরম!", "anger"),
    ("বারবার মিথ্যা বললে কেউ কি ভালো থাকতে পারে?", "anger"),
    ("এই দুর্নীতির কোনো শেষ নেই, অসহ্য লাগছে!", "anger"),
    ("এত বড় ভুল করেও কোনো দুঃখ নেই — মানা যায় না।", "anger"),
    ("রাতে একা থাকলে ভয় লাগে, মনে হয় কেউ আছে।", "fear"),
    ("পরীক্ষার ফলাফলের জন্য ভয়ে বুক কাঁপছে।", "fear"),
    ("ডাক্তারের কাছে যেতে ভয় লাগছে, কী বলবে কে জানে।", "fear"),
    ("নতুন শহরে একা একা ঘুরতে ভয় করছে।", "fear"),
    ("হঠাৎ এই খবর শুনে একদম অবাক হয়ে গেলাম!", "surprise"),
    ("এতদিন পরে তাকে দেখে চমকে উঠলাম।", "surprise"),
    ("এত বড় পুরস্কার আশা করিনি, বিস্ময়ে বাকরুদ্ধ!", "surprise"),
    ("এই ঘটনা সত্যিই ঘটেছে? বিশ্বাস হচ্ছে না!", "surprise"),
    ("এই ধরনের কাজ দেখলে ঘৃণায় মন ভরে যায়।", "disgust"),
    ("মিথ্যাবাদী মানুষের প্রতি আমার তীব্র ঘৃণা আছে।", "disgust"),
    ("এত নোংরা পরিবেশ দেখে গা ঘিনঘিন করছে।", "disgust"),
    ("এরকম অসভ্য আচরণ সত্যিই বিরক্তিকর।", "disgust"),
    ("আজকে বাজারে গিয়েছিলাম, কিছু জিনিস কিনলাম।", "neutral"),
    ("বইটা পড়া হয়ে গেছে, পরের বইটা শুরু করব।", "neutral"),
    ("অফিস থেকে বাড়ি ফিরলাম, রাতের খাবার খেলাম।", "neutral"),
    ("আগামীকাল সকালে মিটিং আছে।", "neutral"),
    ("ট্রেনটা দশ মিনিট দেরিতে এসেছে।", "neutral"),
]


def load_data(csv_path):
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        assert "text" in df.columns and "label" in df.columns
        df = df[df["label"].isin(EMOTIONS)].dropna(subset=["text","label"])
        df["label"] = df["label"].map(LABEL2ID)
        print(f"Loaded {len(df)} rows from {csv_path}")
    else:
        print("Using synthetic demo dataset.")
        df = pd.DataFrame(SAMPLES, columns=["text", "label"])
        df["label"] = df["label"].map(LABEL2ID)
    return df.reset_index(drop=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    round(accuracy_score(labels, preds), 4),
        "f1_weighted": round(f1_score(labels, preds, average="weighted"), 4),
        "f1_macro":    round(f1_score(labels, preds, average="macro"), 4),
    }


def train(csv_path, output_dir, epochs):
    print(f"\n{'='*50}\n  BanglaBERT Emotion Classifier — Fine-tuning\n{'='*50}\n")
    df = load_data(csv_path)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    train_ds = Dataset.from_pandas(train_df).map(tok, batched=True)
    val_ds   = Dataset.from_pandas(val_df).map(tok, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(EMOTIONS),
        id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True,
    )

    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=epochs,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1,
        lr_scheduler_type="cosine", eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f1_weighted", logging_steps=10,
        fp16=torch.cuda.is_available(), report_to="none", save_total_limit=2,
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"\nFinal metrics: {metrics}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(f"{output_dir}/label_config.json", "w", encoding="utf-8") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID,
                   "emotions_bn": EMOTIONS_BN, "eval_metrics": metrics},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {output_dir}/")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   default=None)
    ap.add_argument("--output", default="model_output")
    ap.add_argument("--epochs", type=int, default=5)
    a = ap.parse_args()
    train(a.data, a.output, a.epochs)
