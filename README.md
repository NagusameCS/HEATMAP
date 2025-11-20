# Ollama Heatmap Generator

This tool queries your local Ollama instance with prompts from a CSV file across multiple models and temperature settings. It saves the responses to text files and generates a summary CSV (heatmap) linking to those files.

## Prerequisites

1.  **Python 3**: Ensure Python is installed.
2.  **Ollama**: Must be installed and running on `http://localhost:11434`.
3.  **Models**: Pull the models you want to test using `ollama pull <model_name>`.

## Setup

1.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2.  Prepare your prompts in `prompts.csv`. The format must be:

    ```csv
    id,prompt
    1,Your first prompt here
    2,Your second prompt here
    ```

## Usage

Run the script:

    ```bash
    python heatmap_generator.py
    ```

## Output

-   **Text Responses**: Saved in `output/<model_name>/`.
    -   Filename format: `<model>.t<temp>,num<id>.txt`
-   **Heatmap CSV**: Generated as `heatmap_query_<id>.csv`.
    -   Rows: Models
    -   Columns: Temperatures
    -   Cells: Filenames of the generated responses.