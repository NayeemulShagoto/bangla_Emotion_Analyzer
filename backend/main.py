"""
main.py — FastAPI for BanglaBERT Emotion Analyzer (port 8000)
"""

import io, csv
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List
from inference import BanglaEmotionClassifier

MODEL_PATH = "./model_output"
clf: BanglaEmotionClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global clf
    try:
        clf = BanglaEmotionClassifier(MODEL_PATH)
    except Exception as e:
        print(f"[!] Could not load model: {e}\n    Run `python train.py` first.")
    yield


app = FastAPI(
    title="BanglaBERT Emotion Analyzer",
    description="Fine-tuned BanglaBERT · token heatmap · document timeline",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


def get_clf():
    if clf is None:
        raise HTTPException(503, "Model not loaded. Run `python train.py` first.")
    return clf


# ── Schemas ───────────────────────────────────────────────────────────────────

class AnalyzeReq(BaseModel):
    text: str         = Field(..., min_length=1, max_length=5_000)
    return_tokens: bool = Field(True)

class BatchReq(BaseModel):
    texts: List[str]  = Field(..., min_length=1, max_length=100)

class DocReq(BaseModel):
    text: str         = Field(..., min_length=1, max_length=50_000)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": clf is not None, "backend": "banglabert"}

@app.post("/analyze")
def analyze(req: AnalyzeReq):
    return get_clf().analyze(req.text, return_tokens=req.return_tokens)

@app.post("/analyze/batch")
def analyze_batch(req: BatchReq):
    results = get_clf().analyze_batch(req.texts)
    return {"count": len(results), "results": results}

@app.post("/analyze/document")
def analyze_document(req: DocReq):
    return get_clf().analyze_document(req.text)

@app.post("/analyze/csv")
async def analyze_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only .csv files accepted.")
    content = await file.read()
    reader  = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    if "text" not in (reader.fieldnames or []):
        raise HTTPException(400, "CSV must have a 'text' column.")
    rows = [r for r in reader if r.get("text")][:500]
    results = get_clf().analyze_batch([r["text"] for r in rows])
    emotion_keys = list(results[0]["scores"].keys()) if results else []

    out = io.StringIO()
    fields = (reader.fieldnames or ["text"]) + \
             ["primary_emotion", "emotion_bn", "confidence"] + emotion_keys
    w = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for orig, res in zip(rows, results):
        w.writerow({**orig, "primary_emotion": res["primary_emotion"],
                    "emotion_bn": res["emotion_bn"],
                    "confidence": res["confidence"], **res["scores"]})
    out.seek(0)
    return StreamingResponse(iter([out.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=results.csv"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
