"""
app.py — Gradio frontend for the BanglaBERT Emotion Analyzer.

Calls the FastAPI backend at http://localhost:8000
Run: python app.py   →  http://localhost:7860
"""

import gradio as gr
import requests
import pandas as pd
import tempfile
import os

API = "http://localhost:8000"

EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]

ICONS = {
    "joy": "😊", "sadness": "😢", "anger": "😠",
    "fear": "😨", "surprise": "😲", "disgust": "🤢", "neutral": "😐",
}
COLORS = {
    "joy": "#EF9F27", "sadness": "#378ADD", "anger": "#E24B4A",
    "fear": "#7F77DD", "surprise": "#1D9E75", "disgust": "#D4537E", "neutral": "#888780",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_api() -> bool:
    try:
        return requests.get(f"{API}/health", timeout=3).ok
    except Exception:
        return False


def scores_to_df(scores: dict) -> pd.DataFrame:
    """Convert emotion scores dict to a sorted DataFrame for BarPlot."""
    rows = [
        {"Emotion": f"{ICONS.get(e, '')} {e}", "Score (%)": round(v * 100, 1)}
        for e, v in sorted(scores.items(), key=lambda x: -x[1])
    ]
    return pd.DataFrame(rows)


def build_heatmap_html(tokens: list, primary_emotion: str) -> str:
    """Render token attention weights as a coloured HTML span block."""
    if not tokens:
        return "<p style='color:#aaa; font-size:13px'>Token attention not available for this input.</p>"

    accent = COLORS.get(primary_emotion, "#534AB7")

    html = f"""
    <div style='
        font-family: system-ui, sans-serif;
        background: #fafafa;
        border: 0.5px solid #eee;
        border-radius: 10px;
        padding: 16px 18px;
        line-height: 2.6;
        word-break: break-word;
        font-size: 16px;
    '>
    <p style='font-size:11px; color:#aaa; text-transform:uppercase;
              letter-spacing:0.07em; margin:0 0 12px;'>
        Token attention heatmap
        &nbsp;·&nbsp;
        <span style='color:{accent}'>■</span> warmer = model focused more on this word
    </p>
    """

    for t in tokens:
        w = t["weight"]
        # Warm colour: yellow-orange gradient based on weight
        r = min(255, int(200 + 55 * w))
        g = min(255, int(200 - 80 * w))
        b = min(255, int(200 - 160 * w))
        bg   = f"rgb({r},{g},{b})"
        fw   = "700" if w > 0.75 else ("500" if w > 0.4 else "400")
        size = f"{14 + int(4 * w)}px"
        html += (
            f'<span style="'
            f'background:{bg};'
            f'padding:3px 2px;'
            f'margin:0 2px;'
            f'border-radius:4px;'
            f'font-weight:{fw};'
            f'font-size:{size};'
            f'cursor:default;'
            f'" title="attention weight: {w:.3f}">'
            f'{t["token"]}'
            f'</span> '
        )

    html += "</div>"
    return html


def build_timeline_html(timeline: list) -> str:
    """Render per-sentence emotion timeline as an HTML table."""
    if not timeline:
        return ""
    rows = ""
    for row in timeline:
        e     = row["primary_emotion"]
        color = COLORS.get(e, "#888")
        icon  = ICONS.get(e, "")
        conf  = f"{row['confidence'] * 100:.0f}%"
        text  = row["text"][:80] + ("…" if len(row["text"]) > 80 else "")
        rows += f"""
        <tr>
          <td style='padding:8px 12px; color:#888; font-size:12px; width:36px; text-align:center;'>
            {row['index'] + 1}
          </td>
          <td style='padding:8px 12px; font-size:13px; color:#333;'>{text}</td>
          <td style='padding:8px 12px; text-align:center;'>
            <span style='
              padding:3px 10px; border-radius:99px; font-size:12px;
              background:{color}18; color:{color};
              border:0.5px solid {color}44;
            '>{icon} {e}</span>
          </td>
          <td style='padding:8px 12px; text-align:right; font-size:12px;
                     font-weight:600; color:{color};'>{conf}</td>
        </tr>
        """
    return f"""
    <div style='font-family:system-ui,sans-serif; border:0.5px solid #eee;
                border-radius:10px; overflow:hidden; margin-top:4px;'>
      <table style='width:100%; border-collapse:collapse;'>
        <thead>
          <tr style='background:#f8f8fc; border-bottom:0.5px solid #eee;'>
            <th style='padding:10px 12px; font-size:11px; color:#aaa; font-weight:500;
                       text-transform:uppercase; letter-spacing:0.06em;'>#</th>
            <th style='padding:10px 12px; font-size:11px; color:#aaa; font-weight:500;
                       text-transform:uppercase; letter-spacing:0.06em; text-align:left;'>Sentence</th>
            <th style='padding:10px 12px; font-size:11px; color:#aaa; font-weight:500;
                       text-transform:uppercase; letter-spacing:0.06em;'>Emotion</th>
            <th style='padding:10px 12px; font-size:11px; color:#aaa; font-weight:500;
                       text-transform:uppercase; letter-spacing:0.06em; text-align:right;'>Confidence</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """


def primary_emotion_html(emotion: str, emotion_bn: str, confidence: float) -> str:
    color = COLORS.get(emotion, "#534AB7")
    icon  = ICONS.get(emotion, "")
    return f"""
    <div style='
        font-family: system-ui, sans-serif;
        display: flex; align-items: center; gap: 14px;
        padding: 16px 20px;
        background: {color}10;
        border: 0.5px solid {color}44;
        border-left: 4px solid {color};
        border-radius: 10px;
    '>
      <span style='font-size:36px;'>{icon}</span>
      <div>
        <div style='font-size:22px; font-weight:700; color:{color};'>{emotion}</div>
        <div style='font-size:14px; color:#666; margin-top:2px;'>{emotion_bn}</div>
      </div>
      <div style='margin-left:auto; text-align:right;'>
        <div style='font-size:28px; font-weight:700; color:{color};'>
          {confidence * 100:.0f}%
        </div>
        <div style='font-size:11px; color:#aaa;'>confidence</div>
      </div>
    </div>
    """


# ── API call functions ────────────────────────────────────────────────────────

def fn_analyze(text: str):
    """Single text analysis."""
    if not text.strip():
        return (
            "<p style='color:#aaa'>Enter some Bangla text above.</p>",
            None,
            "<p style='color:#aaa'>—</p>",
            gr.update(visible=False),
        )
    try:
        res  = requests.post(f"{API}/analyze",
                             json={"text": text, "return_tokens": True}, timeout=30)
        res.raise_for_status()
        data = res.json()
    except requests.exceptions.ConnectionError:
        msg = "❌ Cannot connect to backend. Make sure `python main.py` is running on port 8000."
        return msg, None, msg, gr.update(visible=False)
    except Exception as e:
        return f"❌ Error: {e}", None, str(e), gr.update(visible=False)

    scores   = data.get("scores", {})
    scores_df = scores_to_df(scores)
    primary  = data.get("primary_emotion", "")
    heatmap  = build_heatmap_html(data.get("token_weights", []), primary)
    badge    = primary_emotion_html(primary, data.get("emotion_bn", ""), data.get("confidence", 0))

    return badge, scores_df, heatmap, gr.update(visible=True)


def fn_document(text: str):
    """Document timeline analysis."""
    if not text.strip():
        return "<p style='color:#aaa'>Enter a longer Bangla text above.</p>", None
    try:
        res  = requests.post(f"{API}/analyze/document",
                             json={"text": text}, timeout=60)
        res.raise_for_status()
        data = res.json()
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to backend.", None
    except Exception as e:
        return f"❌ {e}", None

    dominant = data.get("dominant", "")
    bn       = data.get("dominant_bn", "")
    n        = data.get("sentence_count", 0)
    scores   = data.get("overall_scores", {})
    scores_df = scores_to_df(scores)
    timeline_html = build_timeline_html(data.get("timeline", []))

    summary = f"""
    <div style='font-family:system-ui,sans-serif; margin-bottom:14px;'>
      <span style='font-size:11px; color:#aaa; text-transform:uppercase;
                   letter-spacing:0.07em;'>Document summary · {n} sentences</span><br>
      <span style='font-size:18px; font-weight:600; color:{COLORS.get(dominant,"#333")};'>
        {ICONS.get(dominant,"")} {dominant}
      </span>
      <span style='font-size:13px; color:#888; margin-left:6px;'>{bn}</span>
    </div>
    {timeline_html}
    """
    return summary, scores_df


def fn_batch(file):
    """CSV batch upload → download results CSV."""
    if file is None:
        return None, "Upload a CSV file with a 'text' column."
    try:
        with open(file.name, "rb") as f:
            res = requests.post(
                f"{API}/analyze/csv",
                files={"file": (os.path.basename(file.name), f, "text/csv")},
                timeout=120,
            )
        res.raise_for_status()
    except requests.exceptions.ConnectionError:
        return None, "❌ Cannot connect to backend."
    except Exception as e:
        return None, f"❌ {e}"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb")
    tmp.write(res.content)
    tmp.close()
    rows = res.content.decode().count("\n") - 1
    return tmp.name, f"✅ Done — {rows} rows analysed. File ready to download."


# ── Custom CSS ────────────────────────────────────────────────────────────────

CSS = """
.gr-button-primary { background: #534AB7 !important; border-color: #534AB7 !important; }
.contain { max-width: 900px !important; }
#header { background: linear-gradient(135deg, #1a1040, #352b7c);
          padding: 24px; border-radius: 12px; margin-bottom: 8px; }
#header h1 { color: #fff; margin: 0 0 4px; font-size: 22px; }
#header p  { color: rgba(255,255,255,0.65); margin: 0; font-size: 13px; }
.tab-nav button { font-size: 13px !important; }
"""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="BanglaBERT Emotion Analyzer") as demo:

    # ── Header ──
    gr.HTML("""
    <div id="header">
      <h1>🧠 BanglaBERT Emotion Analyzer</h1>
      <p>Fine-tuned transformer · 7 emotion classes · token attention heatmap · Bengali text</p>
    </div>
    """)

    with gr.Tabs():

        # ─────────── Tab 1: Single Text ───────────
        with gr.Tab("Single Text"):
            with gr.Row():
                with gr.Column(scale=1):
                    txt_input = gr.Textbox(
                        label="Bangla Text",
                        placeholder="বাংলা টেক্সট লিখুন…\n\nযেমন: আজকে খুব মন খারাপ লাগছে।",
                        lines=5,
                    )
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary")

                    gr.Markdown("**Examples**")
                    gr.Examples(
                        examples=[
                            ["আজকে খুব ভালো লাগছে, সারাদিন মন আনন্দে ভরে আছে!"],
                            ["এত অন্যায় সহ্য করা যাচ্ছে না, রাগে মাথা গরম হয়ে যাচ্ছে!"],
                            ["মনটা আজকে খুব ভারী, কিছুতেই ভালো লাগছে না।"],
                            ["হঠাৎ এই খবর শুনে একদম অবাক হয়ে গেলাম!"],
                            ["রাতে একা থাকলে ভয় লাগে, মনে হয় কেউ আছে।"],
                        ],
                        inputs=txt_input,
                    )

                with gr.Column(scale=1):
                    badge_html   = gr.HTML(label="Detected Emotion")
                    scores_plot  = gr.BarPlot(
                        value=None,
                        x="Emotion", y="Score (%)",
                        title="All emotion scores",
                        color="Emotion",
                        y_lim=[0, 100],
                        height=260,
                        visible=False,
                    )

            heatmap_html_out = gr.HTML(label="Token Attention Heatmap")

            analyze_btn.click(
                fn=fn_analyze,
                inputs=[txt_input],
                outputs=[badge_html, scores_plot, heatmap_html_out,
                         scores_plot],   # last one toggles visibility
            )

        # ─────────── Tab 2: Document Timeline ───────────
        with gr.Tab("Document Timeline"):
            gr.Markdown(
                "Paste a longer Bangla text. Each sentence is analysed separately "
                "and an emotion timeline is shown."
            )
            doc_input = gr.Textbox(
                label="Bangla Document",
                placeholder="দীর্ঘ লেখা বা প্যারাগ্রাফ দিন…",
                lines=8,
            )
            doc_btn = gr.Button("📊 Analyse Document", variant="primary")

            doc_summary   = gr.HTML(label="Timeline")
            doc_scores_plot = gr.BarPlot(
                value=None,
                x="Emotion", y="Score (%)",
                title="Average emotion scores across document",
                color="Emotion",
                y_lim=[0, 100],
                height=240,
            )

            doc_btn.click(
                fn=fn_document,
                inputs=[doc_input],
                outputs=[doc_summary, doc_scores_plot],
            )

        # ─────────── Tab 3: Batch CSV ───────────
        with gr.Tab("Batch CSV"):
            gr.Markdown("""
            Upload a `.csv` file with a **`text`** column.
            The backend analyses each row and returns a downloadable CSV
            with `primary_emotion`, `emotion_bn`, `confidence`, and per-emotion scores appended.

            **Limit:** 500 rows per upload.
            """)
            with gr.Row():
                csv_input  = gr.File(label="Upload CSV", file_types=[".csv"])
                csv_output = gr.File(label="Download Results")
            csv_status = gr.Markdown("")
            csv_btn    = gr.Button("⚙️ Run Batch Analysis", variant="primary")

            csv_btn.click(
                fn=fn_batch,
                inputs=[csv_input],
                outputs=[csv_output, csv_status],
            )

    # ── Footer ──
    gr.HTML("""
    <div style='text-align:center; color:#aaa; font-size:12px; margin-top:24px; padding-top:16px;
                border-top:0.5px solid #eee;'>
      Backend: FastAPI · Model: csebuetnlp/banglabert · Frontend: Gradio
    </div>
    """)


if __name__ == "__main__":
    if not check_api():
        print("\n⚠️  Warning: FastAPI backend not reachable at http://localhost:8000")
        print("   Start it with:  cd ../backend && python main.py\n")
    demo.launch(
    server_port=7860, 
    css=CSS, 
    theme=gr.themes.Soft()
)
