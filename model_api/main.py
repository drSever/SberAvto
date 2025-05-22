from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile, shutil, pandas as pd, json
from pathlib import Path

from preprocessing_script import preprocess_from_raw
from inference           import predict

app = FastAPI(
    title="GA Churn Predictor — CatBoost",
    description="Прогнозирует вероятность конверсии по сессиям Google Analytics",
    version="0.4.1",
)

# ------------------------------------------------------------------ #
#                    REST:  /predict  (RAW CSV)                      #
# ------------------------------------------------------------------ #
@app.post("/predict")
async def predict_endpoint(
    ga_sessions: UploadFile = File(..., description="raw GA-sessions CSV"),
    ga_hits    : UploadFile = File(..., description="raw GA-hits CSV"),
):
    with tempfile.TemporaryDirectory() as tmp:
        sess_path = Path(tmp) / "sessions.csv"
        hits_path = Path(tmp) / "hits.csv"
        for src, dst in [(ga_sessions, sess_path), (ga_hits, hits_path)]:
            with open(dst, "wb") as f:
                shutil.copyfileobj(src.file, f)

        try:
            df    = preprocess_from_raw(sess_path, hits_path)
            proba = predict(df)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return {"rows": len(proba), "probabilities": proba.round(6).tolist()}

# ------------------------------------------------------------------ #
#                         Gradio UI  /ui                             #
# ------------------------------------------------------------------ #
try:
    import gradio as gr
    from gradio import mount_gradio_app
except ImportError:
    gr = None

if gr:
    # ---------- helpers ----------
    def _predict_single(json_str: str) -> float:
        try:
            payload = json.loads(json_str or "{}")
            assert isinstance(payload, dict)
        except Exception:
            raise gr.Error("Введите корректный JSON-объект { ... }")

        df = pd.DataFrame([payload])
        return float(predict(df).iloc[0])

    def _predict_file(file_obj):
        if file_obj is None:
            raise gr.Error("Загрузите CSV-файл")
        df = pd.read_csv(file_obj.name)
        probs = predict(df)
        df["probability"] = probs
        return df.head(50)

    # ---------- layout ----------
    with gr.Blocks(title="GA Churn Predictor") as ui:
        gr.Markdown("<h2 style='text-align:center'>GA Churn Predictor — CatBoost</h2>")

        # ----- Single JSON -------------------------------------------------
        with gr.Tab("Single JSON"):
            with gr.Row():
                json_in = gr.Textbox(lines=10,
                                    label="JSON с признаками (см. пример внизу)",
                                    scale=7)
                with gr.Column(scale=3):
                    prob_out = gr.Number(label="Вероятность", precision=4)
                    gr.Button("Predict").click(
                        _predict_single, inputs=json_in, outputs=prob_out
                    )

        # ----- Batch CSV  (двойной upload) ---------------------------------
        with gr.Tab("Batch CSV"):
            gr.Markdown(
                "Загрузите **ДВА** файла: `ga_sessions.csv` и `ga_hits.csv` "
                "(сырые данные Google Analytics)"
            )
            with gr.Row():
                sess_file = gr.File(file_types=[".csv"], label="ga_sessions.csv")
                hits_file = gr.File(file_types=[".csv"], label="ga_hits.csv")
            df_out = gr.Dataframe(label="Preview (≤50 строк)",
                                interactive=False,
                                wrap=True)

            def _predict_two_files(sess, hits):
                if sess is None or hits is None:
                    raise gr.Error("Нужно выбрать оба файла!")
                df = preprocess_from_raw(sess.name, hits.name)
                df["probability"] = predict(df)
                return df.head(50)

            gr.Button("Predict batch").click(
                _predict_two_files, inputs=[sess_file, hits_file], outputs=df_out
            )

    # монтируем UI
    mount_gradio_app(app, ui, path="/ui")
