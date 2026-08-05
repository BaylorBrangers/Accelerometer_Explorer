# Accelerometer Explorer

Explore, label, and analyze accelerometer signals. Train **custom PyTorch** models (Conv1D, LSTM, autoencoder) or fine-tune **Hugging Face** encoders from a Streamlit UI. Run locally or via Docker with mounted data and artifacts.

## Features

- Upload CSV files or load from a local / container path
- Explore axes and magnitude with interactive Plotly charts
- Sliding-window preparation for neural training
- Train path: **Hugging Face** Hub models or **custom PyTorch** architectures
- Analyze checkpoints and export prediction CSVs
- Docker image with `/data` and `/artifacts` volume mounts

## Expected CSV format

| column | required | notes |
|--------|----------|--------|
| `timestamp` | recommended | numeric time; auto-generated if missing |
| `x`, `y`, `z` | yes | accelerometer channels (aliases like `acc_x` accepted) |
| `label` | for activity training | activity / class name |

A synthetic sample is included at [`data/samples/sample_accel.csv`](data/samples/sample_accel.csv).

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
# or: pip install -r requirements.txt && pip install -e .
```

### Run the Streamlit app

```bash
streamlit run app/Home.py
```

Open http://localhost:8501

### Run tests

```bash
pytest
```

## Docker

Build and run with Compose (mounts `./data` → `/data`, `./artifacts` → `/artifacts`):

```bash
docker compose build
docker compose up
```

Or with plain Docker:

```bash
docker build -t accelerometer-explorer .
docker run --rm -p 8501:8501 \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/artifacts:/artifacts" \
  accelerometer-explorer
```

Then open http://localhost:8501. In **Upload Data**, use path mode with `/data/samples/sample_accel.csv` (or your mounted folder).

### Publishing an image (optional)

```bash
docker tag accelerometer-explorer:latest YOUR_REGISTRY/accelerometer-explorer:0.1.0
docker push YOUR_REGISTRY/accelerometer-explorer:0.1.0
```

Registry CI is not included in this scaffold.

## Project layout

```
src/accel_explorer/   # IO, preprocess, datasets, models, analyze
app/                  # Streamlit multipage UI
data/samples/         # example CSV
artifacts/            # checkpoints & HF cache (Docker mount)
tests/
Dockerfile
docker-compose.yml
```

## Model sources

- **Custom PyTorch**: `Conv1DClassifier`, `LSTMClassifier` for activity; `AccelAutoencoder` for anomaly scores
- **Hugging Face**: load a Hub encoder (presets include `prajjwal1/bert-tiny`, DistilBERT, BERT) wrapped with a window→embedding adapter for fine-tuning

GPU Docker images are out of scope for this scaffold; the default image installs CPU PyTorch.
