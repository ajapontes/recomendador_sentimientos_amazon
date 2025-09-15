# Recomendador de Productos con Sentimiento (Amazon Electronics)

Sistema recomendador **híbrido** que combina señales clásicas (popularidad/CF)
con **análisis de sentimiento** de reseñas usando **DistilBERT** acelerado por **CUDA** (PyTorch).  
La salida se expone como **API** con **FastAPI**.

## Características principales
- **Dataset**: Amazon US Reviews — *Electronics_v1_00*.
- **Sentimiento**: `distilbert-base-uncased-finetuned-sst-2-english` (Transformers).
- **GPU**: Soporte CUDA.
- **API**: FastAPI + Uvicorn.
- **Buenas prácticas**: Clean Code, configuración centralizada en `settings.yaml`.

## Estructura (provisional)
