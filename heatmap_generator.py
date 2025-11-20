import requests
import csv
import os
import json
import sys
from prompt_toolkit.shortcuts import checkboxlist_dialog
from prompt_toolkit.styles import Style
from rich.console import Console, Group
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich import print as rprint

# Configuration
OLLAMA_API_BASE = "http://localhost:11434"
INPUT_CSV = "prompts.csv"
OUTPUT_DIR = "output"
# Generate temperatures from 0.0 to 1.0 with 0.1 step
TEMPERATURES = [round(x * 0.1, 1) for x in range(11)]

console = Console()

def get_available_models():
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/api/tags")
        response.raise_for_status()
        data = response.json()
        # Extract model names. 'models' key contains a list of dicts with 'name'
        return [model['name'] for model in data['models']]
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error connecting to Ollama:[/bold red] {e}")
        console.print("[yellow]Ensure Ollama is running on localhost:11434[/yellow]")
        sys.exit(1)

def create_model_directories(models):
    for model in models:
        # Sanitize model name for directory path (replace : with _)
        safe_name = model.replace(':', '_')
        path = os.path.join(OUTPUT_DIR, safe_name)
        os.makedirs(path, exist_ok=True)

def load_prompts(filepath):
    prompts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'id' in row and 'prompt' in row:
                    prompts.append(row)
                else:
                    console.print(f"[yellow]Warning: Skipping invalid row in CSV: {row}[/yellow]")
    except FileNotFoundError:
        console.print(f"[bold red]Error: Input file '{filepath}' not found.[/bold red]")
        sys.exit(1)
    return prompts

def query_ollama_stream(model, prompt, temp):
    url = f"{OLLAMA_API_BASE}/api/generate"
    # Explicitly set context to empty list to ensure statelessness (no memory of previous prompts)
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temp,
        "stream": True,
        "context": [] 
    }
    try:
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    body = json.loads(line)
                    response_part = body.get('response', '')
                    yield response_part
                    if body.get('done', False):
                        break
    except requests.exceptions.RequestException as e:
        yield f"ERROR: {e}"

def main():
    console.print(Panel.fit("[bold magenta]Ollama Heatmap Generator[/bold magenta]", border_style="cyan"))
    
    # 1. Get Models
    with console.status("[bold green]Fetching available models...[/bold green]", spinner="dots"):
        all_models = get_available_models()
    
    if not all_models:
        console.print("[bold red]No models found in Ollama.[/bold red]")
        sys.exit(1)
    
    # Interactive Selection
    # Using prompt_toolkit's checkboxlist_dialog to satisfy "Enter to select" and "Esc to exit"
    selected_models = checkboxlist_dialog(
        title="Model Selection",
        text="Select models to test:\n(Enter/Space to toggle, Tab to move to OK/Cancel, Esc to exit)",
        values=[(model, model) for model in all_models]
    ).run()

    if selected_models is None:
        console.print("[yellow]Selection cancelled (Esc pressed). Exiting.[/yellow]")
        sys.exit(0)

    if not selected_models:
        console.print("[yellow]No models selected. Exiting.[/yellow]")
        sys.exit(0)

    console.print(f"[green]Selected models:[/green] {', '.join(selected_models)}")

    # 2. Create Directories
    create_model_directories(selected_models)

    # 3. Load Prompts
    prompts = load_prompts(INPUT_CSV)
    console.print(f"[blue]Loaded {len(prompts)} prompts.[/blue]")

    # Data structure to hold results for heatmap:
    # heatmap_data[query_id][model_name][temp] = filename
    heatmap_data = {}

    # 4. Run Queries
    total_ops = len(selected_models) * len(TEMPERATURES) * len(prompts)
    
    # Progress Bar Setup
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
    task_id = progress.add_task("[cyan]Running queries...", total=total_ops)
    
    # Output Panel Setup
    output_panel = Panel("", title="Live Model Output", border_style="blue", height=12)
    
    # Group for Live Display
    display_group = Group(
        progress,
        output_panel
    )

    console.print("[bold yellow]Press Ctrl+C to stop the program at any time.[/bold yellow]")

    try:
        with Live(display_group, console=console, refresh_per_second=10) as live:
            for prompt_data in prompts:
                q_id = prompt_data['id']
                q_text = prompt_data['prompt']
                heatmap_data[q_id] = {}

                for model in selected_models:
                    heatmap_data[q_id][model] = {}
                    safe_model_name = model.replace(':', '_')
                    
                    for temp in TEMPERATURES:
                        progress.update(task_id, description=f"[cyan]Querying {model} (T={temp}) for Prompt {q_id}...")
                        
                        # Stream response
                        full_response = ""
                        error_occurred = False
                        
                        # Update panel title
                        output_panel.title = f"Live Output: {model} (T={temp}) - Prompt {q_id}"
                        output_panel.renderable = "" # Clear previous output
                        
                        for chunk in query_ollama_stream(model, q_text, temp):
                            if chunk.startswith("ERROR:"):
                                error_occurred = True
                                full_response = chunk
                                break
                            full_response += chunk
                            # Update panel with latest content (keep it "tiny" by showing tail if needed, or just full)
                            # Showing last 500 chars to fit in window
                            display_text = full_response[-1000:] 
                            output_panel.renderable = Text(display_text)
                        
                        if not error_occurred:
                            # Naming scheme: llm_name.t05,num01.txt
                            temp_str = f"{int(temp*10):02d}" 
                            filename = f"{safe_model_name}.t{temp_str},num{q_id}.txt"
                            filepath = os.path.join(OUTPUT_DIR, safe_model_name, filename)
                            
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.write(full_response)
                            
                            heatmap_data[q_id][model][temp] = filename
                        else:
                            heatmap_data[q_id][model][temp] = "ERROR"
                            console.print(f"[red]{full_response}[/red]")
                        
                        progress.advance(task_id)
    except KeyboardInterrupt:
        console.print("\n[bold red]Program stopped by user.[/bold red]")
        sys.exit(0)

    # 5. Generate Heatmap CSVs
    console.print("[bold green]Generating heatmap CSVs...[/bold green]")
    for q_id, models_data in heatmap_data.items():
        csv_filename = f"heatmap_query_{q_id}.csv"
        
        # Columns: Model, T=0.1, T=0.3, ...
        fieldnames = ['Model'] + [f"T={t}" for t in TEMPERATURES]
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for model in selected_models:
                row = {'Model': model}
                for temp in TEMPERATURES:
                    row[f"T={temp}"] = models_data.get(model, {}).get(temp, "N/A")
                writer.writerow(row)
        
        console.print(f"Created [bold]{csv_filename}[/bold]")

    console.print(Panel.fit("[bold green]All tasks completed successfully![/bold green]", border_style="green"))

if __name__ == "__main__":
    main()
