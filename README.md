# Ollama Heatmap Generator

A powerful, interactive CLI tool for benchmarking and evaluating Local LLMs using Ollama.

## Features

*   **Interactive TUI**: Navigate easily with a rich terminal user interface.
*   **Benchmark Mode**:
    *   **Standard Mode**: Run models sequentially.
    *   **Crunch Mode**: Run multiple models in parallel (High RAM usage).
    *   **Resume Capability**: Automatically saves progress. Resume interrupted sessions seamlessly.
*   **Evaluation Mode**: Use a "Judge" model to evaluate the quality of responses from other models.
*   **Statistics**: View health checks, storage usage, and session runtimes.
*   **Model Management**: Download, inspect, and browse models directly from the tool.
*   **Audio Feedback**: Satisfying sound effects for process completion and errors.

## Prerequisites

1.  **Python 3.10+**: Ensure Python is installed.
2.  **Ollama**: Must be installed and running on `http://localhost:11434`.
3.  **Models**: You can download models within the tool or use `ollama pull`.

## Setup

1.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2.  Prepare your prompts in `prompts.csv`. The format must be:

    ```csv
    id,prompt
    1,Write a haiku about code.
    2,Explain quantum entanglement like I'm five.
    ```

## Usage

Run the script:

```bash
python heatmap_generator.py
```

### Main Menu Options

*   **Run Benchmark**: Select models and start generating responses.
*   **Resume Previous Session**: Appears if a session was interrupted. Continues where you left off.
*   **Evaluate Sessions**: Use an LLM to grade the generated responses.
*   **View Statistics**: Check disk usage and success rates.
*   **Purge Records**: Clean up old data to free space.
*   **Download/Inspect Models**: Manage your local Ollama library.

## Output Structure

*   `output/`: Contains all generated data.
    *   `<ModelName>/<SessionID>/`: Stores individual response files.
    *   `Summaries/`: Contains generated heatmaps and CSV summaries.
*   `memory.json`: Stores the state of the current/interrupted session.
*   `stats.txt`: Generated report of system usage and benchmarks.

