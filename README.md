# Recomendador de Productos con Sentimiento (Amazon Electronics)

Sistema recomendador **híbrido** para la categoría **Electronics** (Amazon) que combina:
- **Popularidad bayesiana** (ranking global robusto),
- **Filtrado colaborativo por ítems (ItemCF)** (personalización),
- **Análisis de sentimiento** en **texto de reseñas** con **DistilBERT** en **CUDA**.

La salida se expone como **API REST** con **FastAPI**.

---

## 1) ¿Qué es, por qué y cómo?

### Qué
Un servicio que sugiere productos:
- **Globalmente** (lo mejor para todos) y
- **Personalizadamente** (lo mejor para *ti*),  
usando además el **sentimiento** de las reseñas para **promover lo bien valorado en texto** y **penalizar lo popular pero con reseñas negativas**.

### Por qué
- La **popularidad** sola es frágil: puede inflarse con pocos votos o tener críticas negativas.
- El **colaborativo** personaliza, pero puede recomendar “populares pero odiados”.
- El **sentimiento** del texto capta matices que el rating numérico no refleja.

### Cómo (ideas clave)
- **Popularidad bayesiana**: corrige el sesgo de pocos votos mezclando la media del ítem con la media global.
- **Sentimiento (DistilBERT)**: probabilidad de reseña positiva (0–1). Se **agrega por producto**.
- **Híbridos**:
  - Global: `hybrid = α·popularidad + (1−α)·sentimiento`
  - Personalizado: `score_final = score_itemcf · ((1−β)·sent_norm + β)`

---

## 2) Arquitectura (mapa rápido)

```
settings.yaml  →  src/utils/config.py
                   │
                   ▼
              ┌────────────┐
              │  models/   │
              │────────────│
              │ baseline   │ popularidad bayesiana
              │ sentiment  │ DistilBERT (CUDA)
              │ hybrid     │ mezcla pop × sent
              │ itemcf     │ filtrado colaborativo por ítems
              │ hybrid_itemcf │ booster de sentimiento para ItemCF
              │ data_loader│ carga parquet procesados
              │ catalog    │ product_id → title
              └────────────┘
                   │
                   ▼
             src/api/main.py  →  FastAPI (endpoints /recommend/*, /metrics)
```

---

## 3) Dataset y preparación

### Origen
Partimos del archivo local:
```
data/in/amazon_reviews_2015.snappy.parquet
```
(Descargado desde: `https://datasets-documentation.s3.eu-west-3.amazonaws.com/amazon_reviews/amazon_reviews_2015.snappy.parquet`)

> Usamos **Electronics** y generamos un **sample de 100k** reseñas para agilidad.

### Scripts
- **Inspección categorías**  
  ```bash
  python src/data/inspect_categories.py
  ```
- **Preparación local** (filtra Electronics, normaliza y guarda)  
  ```bash
  python src/data/prepare_local_amazon.py
  ```
  Salida esperada:
  - `data/processed/electronics_full.parquet`
  - `data/processed/electronics_sample_100k.parquet`

- **Anotación de sentimiento** (DistilBERT + CUDA; trunca a 512 tokens)  
  ```bash
  python -m src.models.annotate_sentiment
  ```
  Salida esperada:
  - `data/processed/electronics_sample_100k_with_sentiment.parquet`
  - `data/processed/product_sentiment_agg.parquet`  ← **agregado por producto**

---

## 4) Modelos (qué hace cada uno)

- **Popularidad (baseline)** — `src/models/baseline.py`  
  Score bayesiano por `product_id` con columnas: `R` (media), `v` (conteo), `score`.
- **Sentimiento** — `src/models/sentiment.py`  
  Pipeline Transformers (`distilbert-base-uncased-finetuned-sst-2-english`) en GPU si hay CUDA.
- **Híbrido global** — `src/models/hybrid.py`  
  Combina popularidad normalizada y sentimiento agregado por ítem con parámetro `α`.
- **ItemCF** — `src/models/itemcf.py`  
  Similaridad de ítems por co-ocurrencia; recomienda similares a lo que te gustó.
- **Híbrido ItemCF** — `src/models/hybrid_itemcf.py`  
  Repondera ItemCF por sentimiento con parámetro `β`.
- **Catálogo** — `src/models/catalog.py`  
  Enriquecer respuestas con `product_title`.

---

## 5) API (FastAPI)

Arranque local:
```bash
uvicorn src.api.main:app --reload --port 8002
```

### Endpoints principales

- **Salud**
  - `GET /health`

- **Popularidad**
  - `GET /recommend/global?n=10`
  - `GET /recommend/user/{user_id}?n=10` (excluye vistos por el usuario)

- **Híbrido global (Popularidad × Sentimiento)**
  - `GET /recommend/hybrid/global?n=10&alpha=0.7&min_reviews_for_sent=3`
  - `GET /recommend/hybrid/user/{user_id}?n=10&alpha=0.7&min_reviews_for_sent=3`

- **Personalizado**
  - `GET /recommend/itemcf/user/{user_id}?n=10`
  - `GET /recommend/hybrid_itemcf/user/{user_id}?n=10&beta=0.5&min_reviews_for_sent=3`

- **Métricas ligeras**
  - `GET /metrics`  → tiempos/estado por ruta (en memoria)

> **Parámetros clave**:
> - `alpha ∈ [0,1]`: peso de popularidad en el híbrido global (default 0.7).
> - `beta ∈ [0,1]`: atenuación del booster en híbrido ItemCF (0=pleno, 1=sin efecto).
> - `min_reviews_for_sent`: mínimo de reseñas con sentimiento para usar la señal.

---

## 6) Configuración

Archivo: `settings.yaml`
```yaml
project:
  name: recomendador_sentimientos_amazon
random_seed: 42

data:
  raw_dir: "./data/raw"
  processed_dir: "./data/processed"

model:
  sentiment_model_name: "distilbert-base-uncased-finetuned-sst-2-english"
  use_cuda: true

api:
  host: "0.0.0.0"
  port: 8002
```

Loader: `src/utils/config.py`.

---

## 7) Instalación y entorno

Requisitos:
- Python 3.11
- Drivers NVIDIA + CUDA (para GPU)
- PowerShell / Bash

Instalación local (venv):
```bash
python -m venv env
# Windows:
.\env\Scriptsctivate
pip install -r requirements.txt
```

Verificar CUDA:
```bash
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
```

---

## 8) Pruebas

Ejecutar test suite:
```bash
pytest -q
```
(Actualmente en verde: **10 passed**).

---

## 9) Docker (CPU)

**Archivos**:
- `.dockerignore`
- `requirements-docker.txt`
- `Dockerfile` (multi-stage; CPU)

Build:
```bash
docker build -t alfredo/reco-sent-amazon:cpu -f Dockerfile .
```

Run (mapeando puerto; dataset no incluido en la imagen):
```bash
docker run --rm -p 8002:8002 alfredo/reco-sent-amazon:cpu
# Opcional: montar data/ si ya tienes parquet procesados
# docker run --rm -p 8002:8002 -v "$PWD/data:/app/data" alfredo/reco-sent-amazon:cpu
```

Probar:
```bash
curl http://localhost:8002/health
curl "http://localhost:8002/recommend/global?n=5"
```

> Variante **GPU**: pendiente (base `nvidia/cuda`, ruedas cu126, `--gpus all`).

---

## 10) Roadmap

- [x] Preparación de datos (local parquet 2015 → Electronics → sample 100k)
- [x] DistilBERT + CUDA → anotación de sentimiento y agregado por producto
- [x] Popularidad bayesiana + Híbrido global
- [x] ItemCF + Híbrido con booster de sentimiento
- [x] API FastAPI + enriquecimiento de títulos + métricas
- [x] Tests (config, baseline, smoke API)
- [x] Docker (CPU)
- [ ] Docker (GPU)
- [ ] CI/CD (GitHub Actions): lint, tests, build/push imagen
- [ ] Métricas extra (MAP@K, NDCG@K), tuning de hiperparámetros
- [ ] Docs extendidas (OpenAPI tags, ejemplos Postman)

---

## 11) Tips y problemas comunes

- **Windows / HuggingFace symlinks**:  
  Warning por cache sin symlinks. Puedes desactivarlo:
  ```bash
  setx HF_HUB_DISABLE_SYMLINKS_WARNING 1
  ```
- **CUDA False**: revisa que PyTorch cu126 coincida con drivers y que estés en el **venv** correcto.
- **Parquets grandes**: si el full es pesado, trabaja con el **sample 100k** para desarrollo.
- **Faltan títulos**: el catálogo se arma a partir del sample; si no aparecen títulos, revisa `src/models/catalog.py`.

---

## 12) Licencia y créditos

- Reseñas de Amazon: dataset público (usado aquí con fines educativos/demo).
- NLP: Hugging Face Transformers y modelo `distilbert-base-uncased-finetuned-sst-2-english`.
- Frameworks: FastAPI, Uvicorn, PyTorch, scikit-learn, pandas.
