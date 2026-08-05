# Accelerometer Explorer

Explore, label, and analyze accelerometer signals. Train **custom PyTorch** models (Conv1D, LSTM, autoencoder) or fine-tune **Hugging Face** encoders from a Streamlit UI. Run locally or via Docker with mounted data and artifacts.

## Features

- **Hugging Face Hub datasets** — connect an open data repo; files cache to disk (~60GB OK)
- **Streaming training** — windows are read line-by-line from cache; full CSVs are never loaded into RAM
- Upload CSV files or load from a local / container path (Anomark weardata supported)
- Explore axes and magnitude with interactive Plotly charts
- Train path: **Hugging Face** encoders or **custom PyTorch** architectures
- Analyze checkpoints and export prediction CSVs
- Docker image with `/data` and `/artifacts` volume mounts (HF cache under artifacts)

## Hugging Face open data (large corpora)

For multi‑GB / ~60GB datasets, use the **Hugging Face Data** page instead of uploading into the browser:

1. Publish (or use) a Hub **dataset** repo with your CSV / Anomark files.
2. In the app, enter `username/repo`, list files, select which to use.
3. **Prepare cache** downloads once into `HF_HOME` / `artifacts/hf_cache` (disk, not RAM).
4. **Train** streams sliding windows from those cached paths (`StreamingWindowDataset`).

Auth for private repos: set `HF_TOKEN` (or paste a token in the UI). With Docker Compose:

```bash
export HF_TOKEN=hf_xxx
docker compose up
```

Tips for open data on the Hub:

- Prefer splitting into multiple files rather than one 60GB blob when possible (easier parallel cache + selection).
- Parquet on the Hub is even better for columnar reads later; CSV/Anomark streaming is supported today.
- Mount a large volume on `/artifacts` so the Hub cache survives container restarts.

## Expected CSV formats

### Generic

| column | required | notes |
|--------|----------|--------|
| `timestamp` | recommended | numeric time; auto-generated if missing |
| `x`, `y`, `z` | yes | accelerometer channels (aliases like `acc_x` accepted) |
| `label` | for activity training | activity / class name |

### Anomark weardata (`*_acc_weardata.csv`)

Detected automatically from the header:

`RegisterAddress,TimestampMicroseconds,DataElement0`

Rows carry more fields than the header lists. The loader maps:

- `TimestampMicroseconds` → `timestamp` (seconds; thousand separators like `"304,607500"` are stripped)
- first three data elements → `x`, `y`, `z` (accelerometer)
- next triads → `gyro_*` and `aux_*` (kept for exploration)

Sample: [`data/samples/sample_anomark_weardata.csv`](data/samples/sample_anomark_weardata.csv)

### Behavior annotations

Optional interval CSV merged by timestamp. Accepted aliases include
`start`/`start_time`, `end`/`end_time`, and `label`/`behavior`.

Upload the annotation file alongside a single weardata file in the Streamlit **Upload Data** page (or pass an annotation path in path mode).

Synthetic labeled sample: [`data/samples/sample_accel.csv`](data/samples/sample_accel.csv).

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
