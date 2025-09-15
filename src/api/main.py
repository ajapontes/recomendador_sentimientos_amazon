from fastapi import FastAPI

app = FastAPI(title="Recomendador Sentimientos Amazon", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}
