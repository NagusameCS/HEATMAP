import requests
import csv
import os
import json
import sys
import psutil
import msvcrt
import subprocess
import concurrent.futures
import math
import time
import winsound
import datetime
import shutil
import webbrowser
import re
import atexit
import signal
from prompt_toolkit.shortcuts import checkboxlist_dialog, radiolist_dialog, message_dialog, button_dialog, input_dialog
from prompt_toolkit.styles import Style
from rich.console import Console, Group
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich.table import Table
from rich.padding import Padding
from rich import print as rprint
from rich import box

# Configuration
OLLAMA_API_BASE = "http://localhost:11434"
INPUT_CSV = "prompts.csv"
EVALUATOR_PROMPT_FILE = "evaluator_prompt.txt"
OUTPUT_DIR = "output"
# Generate temperatures from 0.0 to 1.0 with 0.1 step
TEMPERATURES = [round(x * 0.1, 1) for x in range(11)]

console = Console()

# Global to track subprocess for cleanup
current_subprocess = None

def cleanup_resources():
    """Cleans up resources (subprocesses, loaded models) on exit."""
    global current_subprocess
    
    # 1. Kill subprocess if running
    if current_subprocess and current_subprocess.poll() is None:
        try:
            current_subprocess.terminate()
            # Give it a moment, then force kill if needed
            try:
                current_subprocess.wait(timeout=2)
            except subprocess.TimeoutExpired:
                current_subprocess.kill()
        except Exception:
            pass

    # 2. Unload all models from Ollama
    try:
        # Get running models
        response = requests.get(f"{OLLAMA_API_BASE}/api/ps")
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            for m in models:
                model_name = m['name']
                # Unload by sending keep_alive=0
                requests.post(f"{OLLAMA_API_BASE}/api/generate", json={
                    "model": model_name,
                    "keep_alive": 0
                })
    except Exception:
        pass

# Register cleanup
atexit.register(cleanup_resources)

def get_system_memory_gb():
    """Returns total system RAM in GB."""
    return round(psutil.virtual_memory().total / (1024**3), 2)

def get_available_models_full():
    """Fetches full model details from Ollama."""
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/api/tags")
        response.raise_for_status()
        data = response.json()
        return data['models']
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error connecting to Ollama:[/bold red] {e}")
        console.print("[yellow]Ensure Ollama is running on localhost:11434[/yellow]")
        sys.exit(1)

def format_size(size_bytes):
    """Formats bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

def check_resource_compatibility(model_size_bytes, system_ram_gb):
    """
    Simple heuristic to check if model fits in RAM.
    Note: This is a rough estimate. LLMs often need VRAM for speed, but can run on RAM.
    """
    model_size_gb = model_size_bytes / (1024**3)
    # Heuristic: If model size is > 90% of total RAM, it's risky.
    if model_size_gb > (system_ram_gb * 0.9):
        return False, f"Model size ({model_size_gb:.2f} GB) is very close to or exceeds total RAM ({system_ram_gb} GB)."
    return True, "Likely fits in RAM."

def play_sound(sound_type="notify"):
    """Plays a Windows system sound."""
    try:
        if sound_type == "notify":
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        elif sound_type == "error":
            winsound.MessageBeep(winsound.MB_ICONHAND)
        elif sound_type == "fail":
            winsound.Beep(400, 200)
            winsound.Beep(200, 400)
        elif sound_type == "delete":
            winsound.Beep(300, 100)
        elif sound_type == "back":
            winsound.Beep(1000, 50)
        elif sound_type == "shutdown":
             winsound.Beep(500, 300)
             time.sleep(0.1)
             winsound.Beep(300, 500)
        elif sound_type == "start":
            winsound.Beep(400, 150)
            winsound.Beep(600, 150)
        elif sound_type == "success":
            winsound.Beep(600, 100)
            winsound.Beep(800, 100)
            winsound.Beep(1000, 100)
        elif sound_type == "jingle":
            winsound.Beep(440, 100)
            winsound.Beep(554, 100)
            winsound.Beep(659, 100)
            winsound.Beep(880, 200)
    except:
        pass

def get_session_id():
    """Generates a unique session ID based on timestamp."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def create_session_directories(models, session_id):
    """Creates directory structure: output/Model/SessionID/"""
    for model in models:
        # Sanitize model name for directory path (replace : with _)
        safe_name = model.replace(':', '_')
        path = os.path.join(OUTPUT_DIR, safe_name, session_id)
        os.makedirs(path, exist_ok=True)
    return session_id

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

def inspect_models_ui(all_models):
    """UI to browse and inspect models."""
    while True:
        # Prepare list for radiolist
        choices = []
        choices.append(("GET_MORE", "🌐 Find more models on Ollama.com"))
        
        for m in all_models:
            name = m['name']
            size = format_size(m['size'])
            choices.append((m, f"{name:<30} {size}"))
        
        # UI Style: Default prompt_toolkit style (Blue background, Grey/White dialog)
        selected_model = radiolist_dialog(
            title="Inspect Models",
            text="Select a model to view details:",
            values=choices
        ).run()

        if selected_model is None:
            play_sound("back")
            return # Go back

        if selected_model == "GET_MORE":
            play_sound("jingle")
            webbrowser.open("https://ollama.com/search")
            continue

        # Show details
        name = selected_model['name']
        size_bytes = selected_model['size']
        details = selected_model.get('details', {})
        
        sys_ram = get_system_memory_gb()
        fits, fit_msg = check_resource_compatibility(size_bytes, sys_ram)
        
        fit_status = "YES" if fits else "WARNING"
        
        info_text = (
            f"Name: {name}\n"
            f"Size: {format_size(size_bytes)}\n"
            f"Family: {details.get('family', 'Unknown')}\n"
            f"Parameter Size: {details.get('parameter_size', 'Unknown')}\n"
            f"Quantization: {details.get('quantization_level', 'Unknown')}\n\n"
            f"System RAM: {sys_ram} GB\n"
            f"Can Run? {fit_status}\n"
            f"({fit_msg})"
        )
        
        # UI Style: Default prompt_toolkit style
        message_dialog(title=f"Details: {name}", text=info_text).run()

def select_models_ui(all_models):
    """UI to select models for benchmarking."""
    sys_ram = get_system_memory_gb()
    choices = []
    for m in all_models:
        name = m['name']
        size_bytes = m['size']
        fits, _ = check_resource_compatibility(size_bytes, sys_ram)
        status_icon = "" if fits else "⚠️ "
        display_text = f"{status_icon}{name:<25} ({format_size(size_bytes)})"
        choices.append((name, display_text))

    # UI Style: Default prompt_toolkit style
    return checkboxlist_dialog(
        title="Model Selection",
        text=f"Select models to test (System RAM: {sys_ram} GB):\n(Enter/Space to toggle, Tab to move to OK/Cancel, Esc to exit)",
        values=choices
    ).run()


def create_download_layout(progress_renderable, output_renderable):
    """
    Creates a layout for the download screen.
    """
    layout = Layout()
    
    # --- Panel 1: Progress Section ---
    progress_panel = Panel(
        progress_renderable,
        title="Download Progress",
        title_align="left",
        border_style="#081D30",
        box=box.DOUBLE,
        padding=(1, 1),
        style="#081D30 on #FFFFFF"
    )

    # --- Panel 2: Log Section ---
    output_panel = Panel(
        output_renderable, 
        title="Ollama Log", 
        title_align="left",
        border_style="#081D30",
        box=box.DOUBLE,
        height=8,
        padding=(0, 1),
        style="#081D30 on #FFFFFF"
    )

    # Content Grid for the Dialog
    grid = Table.grid(expand=True)
    grid.add_row(Text("Downloading Model:", style="#081D30"))
    grid.add_row(Text(" "))
    grid.add_row(progress_panel)
    grid.add_row(Text(" "))
    grid.add_row(output_panel)
    
    # --- Main Dialog Frame ---
    dialog = Panel(
        grid,
        title="[#081D30]Model Downloader[/#081D30]",
        title_align="center",
        box=box.DOUBLE,
        width=80,
        padding=(1, 2),
        style="#081D30 on #FFFFFF", # White background
        border_style="#081D30"
    )
    
    # Shadow Wrapper
    shadow = Padding(dialog, (0, 2, 1, 0), style="on #000088")
    
    # Full Screen Layout
    layout.update(
        Align.center(
            shadow,
            vertical="middle",
            style="on #4444FF" # Navy Blue background
        )
    )
    
    return layout

def download_model_ui():
    """UI to download new models."""
    # UI Style: Default prompt_toolkit style
    model_name = input_dialog(
        title="Download Model",
        text="Enter the name of the model to download (e.g., llama2, mistral):"
    ).run()

    if model_name:
        global current_subprocess
        # Setup Progress
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            expand=True
        )
        task_id = progress.add_task(f"Downloading {model_name}", total=100)
        
        # Initial Output
        output_text = Text("Initializing download...", style="#081D30")
        
        # Layout
        layout = create_download_layout(progress, output_text)
        
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

        max_retries = 5
        retry_count = 0
        success = False

        with Live(layout, console=console, refresh_per_second=20, screen=True) as live:
            while retry_count < max_retries and not success:
                if retry_count > 0:
                    output_text = Text(f"Download failed. Retrying ({retry_count}/{max_retries})...", style="bold red")
                    live.update(create_download_layout(progress, output_text))
                    time.sleep(2)

                try:
                    # Run ollama pull and stream output
                    process = subprocess.Popen(
                        ["ollama", "pull", model_name],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding='utf-8',
                        bufsize=1
                    )
                    current_subprocess = process
                    
                    while True:
                        line = process.stdout.readline()
                        if line == '' and process.poll() is not None:
                            break
                        if line:
                            # Clean ANSI codes
                            clean_line = ansi_escape.sub('', line).strip()
                            if clean_line:
                                # Update text
                                output_text = Text(clean_line, style="#081D30")
                                
                                # Try to parse percentage
                                match = re.search(r'(\d{1,3})%', clean_line)
                                if match:
                                    try:
                                        percent = int(match.group(1))
                                        progress.update(task_id, completed=percent)
                                    except ValueError:
                                        pass
                                
                                live.update(create_download_layout(progress, output_text))
                    
                    if process.returncode == 0:
                        success = True
                    else:
                        retry_count += 1
                        play_sound("fail")
                    current_subprocess = None
                except FileNotFoundError:
                    play_sound("error")
                    message_dialog(title="Error", text="Ollama executable not found. Is it installed and in PATH?").run()
                    return
                except Exception as e:
                    output_text = Text(f"Error: {e}", style="bold red")
                    live.update(create_download_layout(progress, output_text))
                    retry_count += 1
                    play_sound("fail")
            
            if success:
                progress.update(task_id, completed=100, description=f"Downloaded {model_name}")
                play_sound("success")
                message_dialog(title="Success", text=f"Model '{model_name}' downloaded successfully.").run()
            else:
                play_sound("error")
                message_dialog(title="Error", text=f"Failed to download '{model_name}' after {max_retries} attempts.").run()
    else:
        play_sound("back")

def create_layout(progress_renderable, output_renderable):
    """
    Creates a layout mimicking the main menu style (Blue background, White Dialog, Shadow).
    """
    layout = Layout()
    
    # --- Panel 1: Progress Section ---
    progress_panel = Panel(
        progress_renderable,
        title="Data Gathering Progress",
        title_align="left",
        border_style="#081D30",
        box=box.DOUBLE,
        padding=(1, 1),
        style="#081D30 on #FFFFFF"
    )

    # --- Panel 2: Live Output Section ---
    output_panel = Panel(
        output_renderable, 
        title="Live Model Output", 
        title_align="left",
        border_style="#081D30",
        box=box.DOUBLE,
        height=12,
        padding=(0, 1),
        style="#081D30 on #FFFFFF"
    )

    # Content Grid for the Dialog
    grid = Table.grid(expand=True)
    grid.add_row(Text("Current Session Status:", style="#081D30"))
    grid.add_row(Text(" "))
    grid.add_row(progress_panel)
    grid.add_row(Text(" "))
    grid.add_row(output_panel)
    
    # --- Main Dialog Frame ---
    dialog = Panel(
        grid,
        title="[#081D30]Ollama Heatmap Data Gatherer[/#081D30]",
        title_align="center",
        box=box.DOUBLE,
        width=80,
        padding=(1, 2),
        style="#081D30 on #FFFFFF", # White background
        border_style="#081D30"
    )
    
    # Shadow Wrapper
    # Adds a shadow effect using padding with a background color
    # (top, right, bottom, left)
    shadow = Padding(dialog, (0, 2, 1, 0), style="on #000088")
    
    # Full Screen Layout
    layout.update(
        Align.center(
            shadow,
            vertical="middle",
            style="on #4444FF" # Navy Blue background
        )
    )
    
    return layout

def run_benchmark_session(selected_models_names, prompts, crunch_mode=False):
    """
    Runs the benchmark session. 
    If crunch_mode is True, runs in parallel.
    If crunch_mode is False, runs sequentially (max_workers=1).
    """
    
    # RAM Check for Crunch Mode
    if crunch_mode:
        all_models_data = get_available_models_full()
        total_size_bytes = 0
        for m in all_models_data:
            if m['name'] in selected_models_names:
                total_size_bytes += m['size']
        
        total_size_gb = total_size_bytes / (1024**3)
        sys_ram_gb = get_system_memory_gb()
        limit_gb = sys_ram_gb * 0.75
        
        if total_size_gb > limit_gb:
            cont = button_dialog(
                title="RAM Warning",
                text=f"Selected models require {total_size_gb:.2f} GB RAM.\n"
                     f"Limit (75% of System): {limit_gb:.2f} GB.\n\n"
                     "System might become unresponsive.",
                buttons=[("Continue", True), ("Cancel", False)]
            ).run()
            if not cont: return

    # Session Setup
    session_id = get_session_id()
    create_session_directories(selected_models_names, session_id)
    
    heatmap_data = {} 
    for p in prompts:
        heatmap_data[p['id']] = {}
        for m in selected_models_names:
            heatmap_data[p['id']][m] = {}

    # Progress Bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, style="black on white", complete_style="blue", finished_style="green"),
        TaskProgressColumn(text_format="[progress.percentage]{task.percentage:>3.2f}%"),
        TimeElapsedColumn(),
        expand=True
    )
    
    model_tasks = {}
    total_ops_per_model = len(prompts) * len(TEMPERATURES)
    for model in selected_models_names:
        task_id = progress.add_task(f"[cyan]{model}", total=total_ops_per_model)
        model_tasks[model] = task_id

    cancel_event = False
    
    # Output Panel Content (Shared)
    output_content = Text("Initializing...", style="white")
    current_live_model = None

    def process_model(model_name):
        nonlocal cancel_event, output_content, current_live_model
        safe_model_name = model_name.replace(':', '_')
        task_id = model_tasks[model_name]
        
        for prompt_data in prompts:
            if cancel_event: break
            q_id = prompt_data['id']
            q_text = prompt_data['prompt']
            
            for temp in TEMPERATURES:
                if cancel_event: break
                
                progress.update(task_id, description=f"[cyan]{model_name}[/cyan] (T={temp})")
                
                full_response = ""
                error_occurred = False
                
                # In crunch mode, claim the display for this model
                current_live_model = model_name
                
                for chunk in query_ollama_stream(model_name, q_text, temp):
                    if cancel_event: break
                    if chunk.startswith("ERROR:"):
                        error_occurred = True
                        full_response = chunk
                        break
                    full_response += chunk
                    
                    # Show live if standard mode OR if this model owns the display in crunch mode
                    if not crunch_mode or current_live_model == model_name:
                        # Update shared output content
                        output_content = Text(f"{model_name} (T={temp}):\n{full_response[-500:]}", style="green")

                if cancel_event: break

                if not error_occurred:
                    temp_str = f"{int(temp*10):02d}" 
                    filename = f"{safe_model_name}.t{temp_str},num{q_id}.txt"
                    filepath = os.path.join(OUTPUT_DIR, safe_model_name, session_id, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(full_response)
                    heatmap_data[q_id][model_name][temp] = filename
                else:
                    heatmap_data[q_id][model_name][temp] = "ERROR"
                    play_sound("fail")
                
                progress.advance(task_id)

    try:
        workers = len(selected_models_names) if crunch_mode else 1
        mode_text = "Crunch Mode (Parallel)" if crunch_mode else "Standard Mode (Sequential)"
        
        play_sound("start")
        
        with Live(create_layout(progress, output_content), console=console, refresh_per_second=10, screen=True) as live:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(process_model, m) for m in selected_models_names]
                
                while any(f.running() for f in futures):
                    try:
                        # Update layout with current output_content
                        # Note: create_layout takes renderables, so we pass the Text object
                        live.update(create_layout(progress, output_content))
                    except Exception:
                        # Screen crash detection: If update fails, ignore and continue processing
                        pass
                    
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key == b'\x1b':
                            cancel_event = True
                            executor.shutdown(wait=False, cancel_futures=True)
                            play_sound("error")
                            break
                    time.sleep(0.1)
                
                concurrent.futures.wait(futures)

    except KeyboardInterrupt:
        cancel_event = True
        play_sound("error")
    except Exception as e:
        console.print(f"[bold red]Critical Error:[/bold red] {e}")
        play_sound("error")

    play_sound("success")

def purge_records_ui():
    """UI to purge old session records."""
    # 1. List Models
    if not os.path.exists(OUTPUT_DIR):
        # UI Style: Default prompt_toolkit style
        message_dialog(title="Info", text="No output directory found.").run()
        return

    models = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d)) and d != "Summaries"]
    if not models:
        # UI Style: Default prompt_toolkit style
        message_dialog(title="Info", text="No model records found.").run()
        return

    # UI Style: Default prompt_toolkit style
    choices = [("ALL", "🚨 PURGE ALL HISTORY (All Models)")] + [(m, m) for m in models]
    
    selected_model = radiolist_dialog(
        title="Purge Records - Select Model",
        text="Select a model to view sessions, or purge everything:",
        values=choices
    ).run()

    if not selected_model:
        play_sound("back")
        return

    if selected_model == "ALL":
        confirm = button_dialog(
            title="⚠️  DANGER: PURGE ALL  ⚠️",
            text=f"Are you sure you want to delete ALL history for {len(models)} models?\nThis cannot be undone.",
            buttons=[("Yes, DELETE EVERYTHING", True), ("Cancel", False)]
        ).run()
        
        if confirm:
            play_sound("delete")
            for m in models:
                try:
                    shutil.rmtree(os.path.join(OUTPUT_DIR, m))
                except Exception as e:
                    console.print(f"[red]Error deleting {m}: {e}[/red]")
            message_dialog(title="Success", text="All history purged.").run()
        return

    model_path = os.path.join(OUTPUT_DIR, selected_model)
    sessions = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
    
    if not sessions:
        # UI Style: Default prompt_toolkit style
        message_dialog(title="Info", text="No sessions found for this model.").run()
        return

    # UI Style: Default prompt_toolkit style
    selected_sessions = checkboxlist_dialog(
        title=f"Purge Sessions - {selected_model}",
        text="Select sessions to DELETE (Space to select):",
        values=[(s, s) for s in sessions]
    ).run()

    if selected_sessions:
        # UI Style: Default prompt_toolkit style
        confirm = button_dialog(
            title="Confirm Deletion",
            text=f"Are you sure you want to delete {len(selected_sessions)} sessions?",
            buttons=[("Yes, Delete", True), ("No", False)]
        ).run()

        if confirm:
            play_sound("delete")
            for s in selected_sessions:
                shutil.rmtree(os.path.join(model_path, s))
            # UI Style: Default prompt_toolkit style
            message_dialog(title="Success", text="Sessions deleted.").run()

def shutdown_animation():
    """Plays a TV turn-off animation."""
    play_sound("shutdown")
    
    # Force clear screen using system command to ensure clean slate
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Get terminal size explicitly
    cols, rows = shutil.get_terminal_size()
    
    # Shrink vertically
    for i in range(rows // 2):
        console.clear()
        # Draw white bar in center
        h = max(1, rows - (i*2))
        console.print(Align.center(Panel("", style="on white", height=h, width=cols), vertical="middle"))
        time.sleep(0.02)
    
    # Shrink horizontally
    for i in range(cols // 2):
        console.clear()
        w = max(1, cols - (i*2))
        console.print(Align.center(Panel("", style="on white", height=1, width=w), vertical="middle"))
        time.sleep(0.01)
        
    console.clear()
    console.print(Align.center("[bold white]SHUTTING DOWN...[/bold white]", vertical="middle"))
    time.sleep(1)
    console.clear()



def main():
    # 1. Get Models
    with console.status("[bold green]Fetching available models...[/bold green]", spinner="dots"):
        all_models_data = get_available_models_full()
    
    if not all_models_data:
        console.print("[bold red]No models found in Ollama.[/bold red]")
        sys.exit(1)

    # Main Menu Loop
    selected_models_names = []
    
    while True:
        # UI Style: Default prompt_toolkit style
        action = radiolist_dialog(
            title="Ollama Heatmap Data Gatherer",
            text="Choose an action:",
            values=[
                ("benchmark", "Run Benchmark"),
                ("evaluate", "Evaluate Sessions"),
                ("purge", "Purge Old Records"),
                ("inspect", "Browse & Inspect Models"),
                ("download", "Download New Models"),
                ("get_more", "🌐 Find more models on Ollama.com"),
                ("github", "🐙 Visit GitHub Page"),
                ("exit", "Shut Down")
            ]
        ).run()

        if action is None or action == "exit":
            if action is None: play_sound("back")
            shutdown_animation()
            sys.exit(0)
        
        if action == "github":
            play_sound("jingle")
            webbrowser.open("https://github.com/NagusameCS/Heatmap")
            continue

        if action == "get_more":
            play_sound("jingle")
            webbrowser.open("https://ollama.com/search")
            continue
        
        if action == "inspect":
            inspect_models_ui(all_models_data)
            
        if action == "download":
            download_model_ui()
            # Refresh models after download
            with console.status("[bold green]Refreshing models...[/bold green]", spinner="dots"):
                all_models_data = get_available_models_full()
        
        if action == "purge":
            purge_records_ui()

        if action == "evaluate":
            evaluate_sessions_ui(all_models_data)

        if action == "benchmark":
            selected_models_names = select_models_ui(all_models_data)
            if selected_models_names:
                prompts = load_prompts(INPUT_CSV)
                console.print(f"[blue]Loaded {len(prompts)} prompts.[/blue]")
                
                # Ask for Crunch Mode only if multiple models are selected
                crunch_mode = False
                if len(selected_models_names) > 1:
                    # UI Style: Default prompt_toolkit style
                    crunch_mode = button_dialog(
                        title="Benchmark Mode",
                        text="Enable Crunch Mode? (Parallel Processing)\n\n"
                             "Crunch Mode runs all models simultaneously.\n"
                             "Faster, but uses significantly more RAM.",
                        buttons=[("Y/Crunch", True), ("N/Standard", False)]
                    ).run()
                
                run_benchmark_session(selected_models_names, prompts, crunch_mode=crunch_mode)
            else:
                play_sound("back")

    console.print(Panel.fit("[bold green]All tasks completed successfully![/bold green]", border_style="green"))

def load_evaluator_prompt():
    try:
        with open(EVALUATOR_PROMPT_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # Create default if missing
        default_prompt = (
            "You are an impartial evaluator. Categorize the following answer.\n"
            "Return a JSON object with 'category' and 'justification'."
        )
        with open(EVALUATOR_PROMPT_FILE, 'w', encoding='utf-8') as f:
            f.write(default_prompt)
        return default_prompt

def run_evaluation_session(tasks_map, evaluator_model, prompts_map, crunch_mode=False):
    """
    Runs the evaluation session.
    tasks_map: dict { model_name: [list of session_paths] }
    """
    meta_prompt = load_evaluator_prompt()
    
    # Collect all files to process
    file_tasks = []
    for model_name, sessions in tasks_map.items():
        for session_path in sessions:
            if not os.path.exists(session_path): continue
            for fname in os.listdir(session_path):
                if fname.endswith(".txt") and not fname.endswith("_eval.json"):
                    # Parse ID
                    # Format: model.tXX,numID.txt
                    match = re.search(r",num(\d+)\.txt$", fname)
                    if match:
                        q_id = match.group(1)
                        if q_id in prompts_map:
                            file_tasks.append({
                                "path": os.path.join(session_path, fname),
                                "q_id": q_id,
                                "model": model_name,
                                "filename": fname
                            })

    if not file_tasks:
        message_dialog(title="Info", text="No valid files found to evaluate.").run()
        return

    # Progress Bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, style="black on white", complete_style="blue", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        expand=True
    )
    
    task_id = progress.add_task(f"Evaluating with {evaluator_model}", total=len(file_tasks))
    
    cancel_event = False
    output_content = Text("Initializing Evaluation...", style="white")
    current_live_file = None

    def process_file(task_info):
        nonlocal cancel_event, output_content, current_live_file
        if cancel_event: return

        fpath = task_info['path']
        q_id = task_info['q_id']
        original_question = prompts_map[q_id]
        
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                answer_content = f.read()
            
            # Construct Prompt
            full_prompt = (
                f"{meta_prompt}\n\n"
                f"Q: {original_question}\n"
                f"A: {answer_content}"
            )
            
            # Claim display
            current_live_file = task_info['filename']

            # Query Evaluator
            response_text = ""
            for chunk in query_ollama_stream(evaluator_model, full_prompt, 0.1): # Low temp for evaluation
                if cancel_event: break
                response_text += chunk
                if not crunch_mode or current_live_file == task_info['filename']:
                     output_content = Text(f"Evaluating {task_info['filename']}...\n{response_text[-500:]}", style="cyan")

            if cancel_event: return

            # Save Result
            eval_path = fpath.replace(".txt", "_eval.json")
            
            # Try to extract JSON if wrapped in markdown
            json_str = response_text
            json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            
            # Validate JSON (optional, but good practice)
            try:
                # Just ensure it's valid JSON before saving, or save raw if invalid
                json.loads(json_str)
                final_content = json_str
            except json.JSONDecodeError:
                # If invalid, save raw response but maybe wrap it
                final_content = json.dumps({"error": "Invalid JSON", "raw": response_text})

            with open(eval_path, 'w', encoding='utf-8') as f:
                f.write(final_content)
                
        except Exception as e:
            if not crunch_mode:
                output_content = Text(f"Error evaluating {task_info['filename']}: {e}", style="red")

        progress.advance(task_id)

    # Execution
    try:
        workers = 4 if crunch_mode else 1 # Cap workers for evaluation to avoid OOM if evaluator is large
        play_sound("start")
        
        with Live(create_layout(progress, output_content), console=console, refresh_per_second=10, screen=True) as live:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(process_file, t) for t in file_tasks]
                
                while any(f.running() for f in futures):
                    try:
                        live.update(create_layout(progress, output_content))
                    except: pass
                    
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key == b'\x1b':
                            cancel_event = True
                            executor.shutdown(wait=False, cancel_futures=True)
                            play_sound("error")
                            break
                    time.sleep(0.1)
                
                concurrent.futures.wait(futures)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        play_sound("error")

    play_sound("success")
    message_dialog(title="Success", text="Evaluation complete.").run()

def evaluate_sessions_ui(all_models):
    """UI to setup evaluation session."""
    # 1. Gather all available sessions from disk
    if not os.path.exists(OUTPUT_DIR):
        message_dialog(title="Info", text="No output directory found.").run()
        return

    available_sessions = [] 
    
    # Iterate over directories in OUTPUT_DIR
    for model_dir in os.listdir(OUTPUT_DIR):
        model_path = os.path.join(OUTPUT_DIR, model_dir)
        if os.path.isdir(model_path) and model_dir != "Summaries":
            for session_id in os.listdir(model_path):
                session_path = os.path.join(model_path, session_id)
                if os.path.isdir(session_path):
                    available_sessions.append({
                        "model": model_dir,
                        "session": session_id,
                        "path": session_path
                    })
    
    if not available_sessions:
        message_dialog(title="Info", text="No sessions found.").run()
        return

    # Sort by model then session
    available_sessions.sort(key=lambda x: (x['model'], x['session']))

    # 2. Prepare choices for CheckboxList
    choices = []
    choices.append(("ALL", " [ SELECT ALL SESSIONS ]"))
    
    for s in available_sessions:
        display = f"{s['model']} | {s['session']}"
        value = s['path']
        choices.append((value, display))

    # 3. Show Dialog
    selected_values = checkboxlist_dialog(
        title="Select Sessions to Evaluate",
        text="Pick sessions (grouped by model).\nSelect 'SELECT ALL SESSIONS' to include everything.",
        values=choices
    ).run()

    if not selected_values:
        return

    # Process selection
    final_session_paths = []
    if "ALL" in selected_values:
        final_session_paths = [s['path'] for s in available_sessions]
    else:
        final_session_paths = selected_values

    # Group by model for the runner
    tasks_map = {}
    for path in final_session_paths:
        parent_dir = os.path.dirname(path)
        model_name = os.path.basename(parent_dir)
        
        if model_name not in tasks_map:
            tasks_map[model_name] = []
        tasks_map[model_name].append(path)

    if not tasks_map:
        return

    # 4. Select Evaluator Model
    choices = [(m['name'], f"{m['name']} ({format_size(m['size'])})") for m in all_models]
    evaluator_model = radiolist_dialog(
        title="Select Evaluator Model",
        text="Choose the model that will perform the evaluation:",
        values=choices
    ).run()
    
    if not evaluator_model: return

    # 5. Crunch Mode?
    crunch_mode = button_dialog(
        title="Evaluation Mode",
        text="Enable Crunch Mode? (Parallel Processing)\n\n"
             "Warning: Running multiple instances of the evaluator model requires significant RAM/VRAM.",
        buttons=[("Y/Crunch", True), ("N/Standard", False)]
    ).run()

    # Load Prompts for Q mapping
    prompts_list = load_prompts(INPUT_CSV)
    prompts_map = {p['id']: p['prompt'] for p in prompts_list}

    run_evaluation_session(tasks_map, evaluator_model, prompts_map, crunch_mode)

if __name__ == "__main__":
    print("Starting Heatmap Data Gatherer...")
    try:
        main()
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
        import traceback
        traceback.print_exc()

