"""
inference.py — Load fine-tuned BanglaBERT and run emotion inference.
"""

import re, json
import torch
import numpy as np
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification

EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]

EMOTIONS_BN = {
    "joy": "আনন্দ", "sadness": "দুঃখ", "anger": "রাগ",
    "fear": "ভয়", "surprise": "বিস্ময়", "disgust": "ঘৃণা", "neutral": "নিরপেক্ষ",
}

EMOTION_COLORS = {
    "joy": "#EF9F27", "sadness": "#378ADD", "anger": "#E24B4A",
    "fear": "#7F77DD", "surprise": "#1D9E75", "disgust": "#D4537E", "neutral": "#888780",
}


class BanglaEmotionClassifier:

    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device).eval()

        try:
            cfg = json.load(open(f"{model_path}/label_config.json", encoding="utf-8"))
            self.id2label = {int(k): v for k, v in cfg["id2label"].items()}
        except FileNotFoundError:
            self.id2label = {i: e for i, e in enumerate(EMOTIONS)}

        print(f"Model ready on {self.device}")

    def _encode(self, text: str):
        return self.tokenizer(
            text, return_tensors="pt", truncation=True,
            padding=True, max_length=128,
        )

    def _attention_weights(self, input_ids, attention_mask, attentions) -> List[dict]:
        """Average CLS-row attention across all layers & heads → token weights."""
        stacked = torch.stack(attentions, dim=0)       # (L, 1, H, S, S)
        avg     = stacked.mean(0).mean(1)[0, 0, :]     # (S,)
        tokens  = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        mask    = attention_mask[0].cpu().tolist()

        weights = [
            {"token": t.replace("##", ""), "weight": float(w)}
            for t, w, m in zip(tokens, avg.cpu().numpy(), mask)
            if m == 1 and t not in ("[CLS]", "[SEP]", "[PAD]")
        ]
        mx = max((r["weight"] for r in weights), default=1)
        if mx > 0:
            for r in weights:
                r["weight"] = round(r["weight"] / mx, 4)
        return weights

    # ── Public ────────────────────────────────────────────────────────────────

    def analyze(self, text: str, return_tokens: bool = True) -> dict:
        inp    = self._encode(text)
        ids    = inp["input_ids"].to(self.device)
        mask   = inp["attention_mask"].to(self.device)

        with torch.no_grad():
            out = self.model(ids, attention_mask=mask,
                             output_attentions=return_tokens)

        probs  = torch.softmax(out.logits[0], -1).cpu().numpy()
        scores = {self.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)}
        top    = max(scores, key=scores.get)

        result = {
            "text":             text,
            "primary_emotion":  top,
            "emotion_bn":       EMOTIONS_BN.get(top, top),
            "confidence":       round(scores[top], 4),
            "scores":           scores,
            "emotion_colors":   EMOTION_COLORS,
            "model":            "banglabert-finetuned",
        }
        if return_tokens and out.attentions:
            result["token_weights"] = self._attention_weights(ids, mask, out.attentions)
        return result

    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> List[dict]:
        results = []
        for i in range(0, len(texts), batch_size):
            for t in texts[i:i + batch_size]:
                results.append(self.analyze(t, return_tokens=False))
        return results

    def analyze_document(self, text: str) -> dict:
        chunks = [c.strip() for c in re.split(r"[।!?\.]+", text) if c.strip()]
        if not chunks:
            return self.analyze(text)

        timeline = []
        for i, chunk in enumerate(chunks):
            r = self.analyze(chunk, return_tokens=False)
            timeline.append({
                "index": i, "text": chunk,
                "primary_emotion": r["primary_emotion"],
                "emotion_bn":      r["emotion_bn"],
                "confidence":      r["confidence"],
                "scores":          r["scores"],
            })

        agg = {e: sum(row["scores"].get(e, 0) for row in timeline) / len(timeline)
               for e in EMOTIONS}
        dominant = max(agg, key=agg.get)

        return {
            "text":           text,
            "sentence_count": len(chunks),
            "dominant":       dominant,
            "dominant_bn":    EMOTIONS_BN.get(dominant, dominant),
            "overall_scores": {k: round(v, 4) for k, v in agg.items()},
            "timeline":       timeline,
            "emotion_colors": EMOTION_COLORS,
        }
