<img width="1280" height="640" alt="IRIS-MD (1)" src="https://github.com/user-attachments/assets/9a731931-a919-48c1-a612-9a21faf3fdd3" />
<img src="https://img.shields.io/badge/Python%20project-black?style=for-the-badge&logo=python" alt="Badge">
<img src="https://img.shields.io/badge/GH--Page-0350aa?style=for-the-badge&logo=github&link=https%3A%2F%2Fnagusamecs.github.io%2FHEATMAP%2F" alt="Badge">

Overview
--------
HEATMAP is a lightweight, local tool to benchmark LLM models served via Ollama (or similar local API), aggregate evaluation JSONs, and produce CSV heatmaps (model × temperature) and derivatives (d/dT). The project is targeted at researchers and engineers who want reproducible local benchmarking, simple evaluator aggregation, and CSV outputs for downstream analysis.

Key features
------------
- Cross-platform (macOS/Linux/Windows) audio and keyboard fallbacks.
- Dedicated launcher (`run.py`) which opens the generator in a new terminal and exits the launcher window.
- Single-instance protection via per-user PID file.
- Benchmark runner: multi-model, multi-temperature testing with optional "Crunch Mode" (parallel execution).
- Aggregation: scans `output/` for evaluator JSONs, produces `output/Summaries/super_valid.json`, `invalid_files.json`, and timestamped heatmap CSVs (`heatmap_<id>.csv`).
- CSV index: `output/Summaries/csv_index.json` registers heatmap CSVs with short IDs for selection.
- Derivative generator: computes d/dT across temperature axis and writes `heatmap_derivative_<id>.csv`.
- Settings UI to control temperature axis and question range; persisted to `settings.json`.
- Build helper (`build.py`) and GitHub Actions workflow for producing release zips.

Quick start
-----------
Prerequisites
- Python 3.10 or later
- `ollama` (if you plan to run local models)
- Recommended: create a virtualenv and install dependencies from `requirements.txt`

Run locally (recommended):

```bash
python run.py
```

This opens a dedicated terminal session for the generator and closes the launcher window. Alternatively, run the generator directly:

```bash
python heatmap_generator.py
```

Primary menu overview
---------------------
- Resume / Run Benchmark — start or resume a benchmark session.
- Analysis → Aggregate JSONs & Create Heatmap — scans `output/` for evaluator JSONs and writes a timestamped heatmap CSV.
- Analysis → Generate Heatmap Derivative — pick a heatmap CSV (by ID) and write a derivative CSV.
- Maintenance → Settings — edit temperature min/max/step and question range (persisted to `settings.json`).
- Maintenance → Open Output Folder — opens `output/` in your system file browser.
- More → GitHub / Find more models — helpful links.

Settings
--------
`settings.json` is saved in the project root and contains runtime preferences, for example:

```json
{
  "temp_min": 0.0,
  "temp_max": 1.0,
  "temp_step": 0.1,
  "question_range": {"start": 1, "end": null}
}
```

- `temp_min`, `temp_max`, `temp_step`: control the temperature axis used in benchmarks and heatmap aggregation. The Settings UI regenerates the `TEMPERATURES` used by the app.
- `question_range`: 1-based start/end indices to limit which prompts from `prompts.csv` are used during a benchmark. If `end` is `null`, all prompts from `start` onward are used.

Prompts CSV
-----------
The input prompts file is `prompts.csv` (default). It should include columns `id` and `prompt` at minimum. Example:

```
id,prompt
1,What is the capital of France?
2,Summarize the following paragraph...
```

Outputs and Summaries
---------------------
All session outputs and derived files are saved to `output/`.

Important files under `output/Summaries/`:
- `super_valid.json` — aggregated, successfully parsed evaluator JSONs.
- `invalid_files.json` — JSON files that could not be parsed.
- `heatmap_<sessionid>.csv` — timestamped heatmap CSVs created by aggregation. Columns: `Model, <temp1>, <temp2>, ...` cells contain average accuracy (0–1) or empty if insufficient data.
- `csv_index.json` — index mapping short `id` → metadata for each generated heatmap CSV. Metadata includes `id`, `filename`, `path`, `created_at`, and `temps`.
- `heatmap_derivative_<id>.csv` — derivative outputs, named to include the source CSV id when available.

Aggregation & CSV IDs
---------------------
When you run Analysis → Aggregate JSONs & Create Heatmap, the tool creates a new `heatmap_<sessionid>.csv` and registers it in `csv_index.json` with a short 8-hex id. Use Analysis → Generate Heatmap Derivative to select the exact CSV by ID.

Derivative computation
----------------------
The derivative generator computes numeric derivatives d/dT across the temperature columns using central differences when neighbors are present, and forward/backward differences at the edges. Missing values are preserved as empty cells.

Launcher and single-instance behavior
------------------------------------
- `run.py` opens the generator in a dedicated terminal window and then exits the launcher process when successful.
- The generator uses a PID file in the system temp folder (`/tmp/heatmap_generator_<user>.pid`) to prevent multiple instances by the same user.

Packaging & CI
--------------
- `build.py` orchestrates PyInstaller builds for supported platforms and produces per-target zip artifacts under `dist/`.
- A GitHub Actions workflow (see `.github/workflows/release-build.yml`) builds release artifacts on tagged releases.

Troubleshooting
---------------
- "Ollama executable not found": ensure `ollama` is installed and on your `PATH`.
- Blank heatmap columns: ensure your evaluator writes measurable accuracy (0–1) or categorical labels recognized by the aggregator (e.g., "factual" → 1.0, "incoherent" → 0.0).
- If the UI freezes during benchmarking, press `Esc` to cancel (the app saves progress to `memory.json`).

Development notes
-----------------
- The main app is `heatmap_generator.py`. The launcher is `run.py`.
- Unit tests are not included in Beta 1.0; adding CI tests is planned.
- If you modify the temperature settings manually in `settings.json`, restart the app or re-open Settings to apply them.
