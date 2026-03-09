# Cymatic Seal

Adversarial audio protection for creators. Cymatic Seal renders your music **mathematically un-trainable** by AI source-separation and voice-cloning models while keeping the perturbations completely inaudible to human listeners.

## How It Works

1. **Psychoacoustic analysis** — A full-spectrum masking model (10 ms sliding FFT window, simultaneous + temporal masking) determines, for every time-frequency tile, the maximum perturbation amplitude that stays below the threshold of human audibility.

2. **Gradient-guided perturbation** — The engine loads a source-separation model (Demucs HTDemucs by default) and runs FGSM or I-FGSM to compute the perturbation direction that *maximises* the model's separation error, then projects it onto the psychoacoustic bound.

3. **Sealed output + certificate** — The perturbed audio is saved alongside a Forensic Seal Certificate (JSON) that records which frequency bands were masked, perturbation statistics, and a SHA-256 hash of the sealed file for later verification.

When an AI pipeline tries to separate vocals from a sealed track, it gets hopelessly corrupted stems — making the file useless as training data for voice cloning or style replication.

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

> Requires Python 3.10+. A CUDA-capable GPU is recommended but not required — the engine works on CPU (slower).

### CLI

**Seal a track:**

```bash
python -m cli.main seal my_track.wav --artist "Producer Name" --title "My Song"
```

This creates `my_track_sealed.wav` and `my_track_sealed.wav.cert.json`.

**Verify a sealed track:**

```bash
python -m cli.main verify my_track_sealed.wav my_track_sealed.wav.cert.json
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--method` | `ifgsm` | `fgsm` (fast, 1 step) or `ifgsm` (iterative, stronger) |
| `--steps` | `5` | Number of I-FGSM iterations |
| `--epsilon` | `0.02` | Maximum perturbation amplitude (linear) |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--model` | `htdemucs` | Demucs model name |
| `--targets` | `vocals` | Source stems to disrupt (space-separated) |
| `--margin-db` | `-1.0` | Safety margin below masking threshold (more negative = quieter) |

### Web App

Start the server:

```bash
uvicorn api.app:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

- **Seal page** — Upload an audio file, configure options, download the sealed version + certificate.
- **Verify page** — Upload a sealed file and its certificate to confirm the seal is intact.

### Docker

```bash
docker build -t cymatic-seal .
docker run -p 8000:8000 cymatic-seal
```

## Project Structure

```
cymatic_seal/
  audio/          Audio I/O (load, save, resample, normalize)
  psychoacoustics/ FFT-based masking model (simultaneous + temporal)
  adversarial/    Demucs wrapper, FGSM/I-FGSM engine, epsilon projection
  seal/           Pipeline orchestration, certificate generation
  verify/         Certificate parsing and hash verification
api/              FastAPI web service
frontend/         HTML/CSS/JS UI (seal + verify pages)
cli/              Command-line interface
tests/            Unit and integration tests
```

## Running Tests

```bash
# Fast unit tests (no Demucs model download required)
pytest tests/ -v --ignore=tests/test_integration.py

# Full integration tests (downloads Demucs model, needs ~2 GB)
pytest tests/test_integration.py -v -s
```

## Configuration for Free / Low-Cost Use

The engine is designed to be accessible:

- Use `--device cpu` to avoid needing a GPU.
- Use `--method fgsm --steps 1` for the fastest processing.
- The web app has a 100 MB upload limit by default.
- All processing is local — no external API calls, no data sent anywhere.

## License

MIT
