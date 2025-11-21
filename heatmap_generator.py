import requests
import csv
import os
import json
import sys
import psutil
try:
    import msvcrt
except Exception:
    # Provide a minimal msvcrt-like fallback for Unix (macOS/Linux)
    import sys
    import select
    import tty
    import termios

    class _MsvcrtFallback:
        def kbhit(self):
            try:
                dr, _, _ = select.select([sys.stdin], [], [], 0)
                return bool(dr)
            except Exception:
                return False

        def getch(self):
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                return ch.encode()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    msvcrt = _MsvcrtFallback()
import subprocess
import concurrent.futures
import math
import time
import platform
import tempfile
import wave
import struct
try:
    import winsound
except Exception:
    winsound = None
import datetime
import shutil
import webbrowser
import re
import atexit
import signal
import tempfile
import getpass
import argparse
import shlex
import statistics
import uuid
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
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
STATS_FILE = "stats.txt"
MEMORY_FILE = "memory.json"
SETTINGS_FILE = "settings.json"
# Generate temperatures from 0.0 to 1.0 with 0.1 step
TEMPERATURES = [round(x * 0.1, 1) for x in range(11)]

console = Console()

# Global to track subprocess for cleanup
current_subprocess = None

def save_memory(state):
    """Saves the current state to memory.json."""
    try:
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4)
    except Exception:
        pass

def load_memory():
    """Loads state from memory.json if it exists."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None

def clear_memory():
    """Deletes memory.json."""
    if os.path.exists(MEMORY_FILE):
        try:
            os.remove(MEMORY_FILE)
        except Exception:
            pass

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


# Single-instance lock (PID file) -------------------------------------------------
# Use a per-user pid file in the temp directory so multiple users can run independently.
PID_FILE = os.path.join(tempfile.gettempdir(), f'heatmap_generator_{getpass.getuser()}.pid')

def _is_pid_running(pid):
    try:
        return psutil.pid_exists(int(pid))
    except Exception:
        return False

def acquire_instance_lock():
    """Ensure only one instance runs per user. Exits the process if another instance is active."""
    # If pid file exists, check if process alive
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r', encoding='utf-8') as f:
                existing = f.read().strip()
            if existing:
                try:
                    existing_pid = int(existing)
                    if _is_pid_running(existing_pid) and existing_pid != os.getpid():
                        console.print(f"[red]Another HEATMAP instance is running (PID {existing_pid}). Exiting.[/red]")
                        sys.exit(1)
                except ValueError:
                    pass
        except Exception:
            pass

    # Write our PID
    try:
        with open(PID_FILE, 'w', encoding='utf-8') as f:
            f.write(str(os.getpid()))
    except Exception:
        pass

    # Ensure removal on exit
    def _remove_pidfile():
        try:
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)
        except Exception:
            pass

    atexit.register(_remove_pidfile)

# End single-instance lock -------------------------------------------------------

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
    # Heuristic: If model size is > 80% of total RAM, it's risky.
    if model_size_gb > (system_ram_gb * 0.8):
        return False, f"Model size ({model_size_gb:.2f} GB) is very close to or exceeds total RAM ({system_ram_gb} GB)."
    return True, "Likely fits in RAM."

def _play_tone_winsound(freq, ms):
    try:
        if winsound:
            winsound.Beep(int(freq), int(ms))
    except Exception:
        pass

def _play_tone_afplay(freq, ms, volume=0.5):
    """Generate a short WAV tone and play it with `afplay` (macOS)."""
    try:
        framerate = 44100
        n_samples = int(framerate * (ms / 1000.0))
        amplitude = int(32767 * max(0.0, min(volume, 1.0)))

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        tmpname = tmp.name
        tmp.close()

        with wave.open(tmpname, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(framerate)
            for i in range(n_samples):
                t = float(i) / framerate
                sample = amplitude * math.sin(2 * math.pi * freq * t)
                wf.writeframes(struct.pack('<h', int(sample)))

        # Play with afplay (macOS)
        try:
            subprocess.run(['afplay', tmpname], check=False)
        except Exception:
            pass
        try:
            os.remove(tmpname)
        except Exception:
            pass
    except Exception:
        pass

def play_tone(freq, ms, volume=0.5):
    """Cross-platform single-tone player: Windows uses winsound, macOS uses afplay."""
    try:
        if platform.system() == 'Windows' and winsound:
            _play_tone_winsound(freq, ms)
        elif platform.system() == 'Darwin':
            _play_tone_afplay(freq, ms, volume=volume)
        else:
            # Fallback: try winsound if available, otherwise no-op
            if winsound:
                _play_tone_winsound(freq, ms)
    except Exception:
        pass

def play_sound(sound_type="notify"):
    """Plays a short musical phrase using platform-appropriate tone playback."""
    try:
        # Frequencies for C Major Scale (C4 - C6)
        C4, D4, E4, F4, G4, A4, B4 = 261, 293, 329, 349, 392, 440, 493
        C5, D5, E5, F5, G5, A5, B5 = 523, 587, 659, 698, 784, 880, 987
        C6 = 1046

        if sound_type == "notify":
            # Shorten notify tones to make the notification feel snappier
            play_tone(C5, 40)
            play_tone(E5, 40)

        elif sound_type == "error":
            play_tone(C4, 200)
            play_tone(370, 400)

        elif sound_type == "fail":
            play_tone(G4, 150)
            play_tone(F4, 150)
            play_tone(E4, 150)
            play_tone(D4, 300)

        elif sound_type == "delete":
            play_tone(G4, 50)
            play_tone(C4, 100)

        elif sound_type == "back":
            play_tone(C5, 50)

        elif sound_type == "shutdown":
            play_tone(G4, 150)
            play_tone(E4, 150)
            play_tone(C4, 400)

        elif sound_type == "start":
            play_tone(C4, 100)
            play_tone(E4, 100)
            play_tone(G4, 100)
            play_tone(C5, 200)

        elif sound_type == "success":
            play_tone(C5, 80)
            play_tone(E5, 80)
            play_tone(G5, 80)
            play_tone(C6, 200)

        elif sound_type == "jingle":
            play_tone(C5, 150)
            play_tone(G4, 150)
            play_tone(A4, 150)
            play_tone(E5, 300)

        elif sound_type == "complete":
            play_tone(G4, 100)
            play_tone(C5, 300)

    except Exception:
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


def open_output_dir(path=None):
    """Open a folder in the system file browser (Finder/Explorer/Files).

    If `path` is None, opens the project's `OUTPUT_DIR`.
    """
    target = path or OUTPUT_DIR
    try:
        if not os.path.exists(target):
            message_dialog(title="Info", text=f"Directory '{target}' not found.").run()
            return
    except Exception:
        # If message_dialog isn't available for some reason, fallback to print
        try:
            print(f"Directory '{target}' not found.")
        except Exception:
            pass
        return

    try:
        system = platform.system()
        if system == 'Darwin':
            subprocess.run(['open', target], check=False)
        elif system == 'Windows':
            try:
                os.startfile(target)
            except Exception:
                subprocess.run(['explorer', target], check=False)
        else:
            # Linux/Other: prefer xdg-open, fallback to webbrowser
            try:
                subprocess.run(['xdg-open', target], check=False)
            except Exception:
                webbrowser.open('file://' + os.path.abspath(target))
    except Exception:
        try:
            webbrowser.open('file://' + os.path.abspath(target))
        except Exception:
            message_dialog(title="Error", text=f"Could not open directory: {target}").run()


def _register_csv_entry(summaries_dir, csv_filename, temps):
    """Register a generated CSV in `csv_index.json` with a short id and metadata."""
    index_path = os.path.join(summaries_dir, 'csv_index.json')
    try:
        if os.path.exists(index_path):
            with open(index_path, 'r', encoding='utf-8') as f:
                index = json.load(f)
        else:
            index = {}
    except Exception:
        index = {}

    csv_id = uuid.uuid4().hex[:8]
    entry = {
        'id': csv_id,
        'filename': os.path.basename(csv_filename),
        'path': os.path.abspath(csv_filename),
        'created_at': datetime.datetime.now().isoformat(),
        'temps': [float(t) for t in temps]
    }
    index[csv_id] = entry

    try:
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
    except Exception:
        pass

    return csv_id, entry


def load_settings():
    """Load settings from `settings.json` in project root. Returns dict."""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    # defaults
    return {
        'temp_min': 0.0,
        'temp_max': 1.0,
        'temp_step': 0.1,
        'question_range': {'start': 1, 'end': None}
    }


def save_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass


def apply_temperature_settings(settings):
    """Regenerate global TEMPERATURES from settings dict."""
    global TEMPERATURES
    try:
        tmin = float(settings.get('temp_min', 0.0))
        tmax = float(settings.get('temp_max', 1.0))
        step = float(settings.get('temp_step', 0.1))
        if step <= 0:
            return
        temps = []
        cur = tmin
        # guard against infinite loops
        max_iters = 1000
        it = 0
        while cur <= tmax + 1e-9 and it < max_iters:
            temps.append(round(cur, 4))
            cur = cur + step
            it += 1
        if temps:
            # normalize small floats to 1 decimal if step is multiple of 0.1
            TEMPERATURES = [round(t, 1) if abs(round(step, 1) - step) < 1e-9 else round(t, 4) for t in temps]
    except Exception:
        pass

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

def load_categories():
    cats = {}
    try:
        with open('prompts_categories.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'id' in row and 'category' in row:
                    cats[row['id']] = row['category']
    except Exception:
        pass
    return cats

def query_ollama_stream(model, prompt, temp, json_mode=False):
    url = f"{OLLAMA_API_BASE}/api/generate"
    # Explicitly set context to empty list to ensure statelessness (no memory of previous prompts)
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temp,
        "stream": True,
        "context": [] 
    }
    if json_mode:
        payload["format"] = "json"
        
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
        choices.append(("GET_MORE", "Find more models on Ollama.com"))
        
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
            SpinnerColumn(spinner_name="dots12", style="bold cyan"),
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=None, style="dim white", complete_style="bold cyan", finished_style="bold green"),
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
        title="[#081D30]HEATMAP Data Gatherer[/#081D30]",
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

def run_benchmark_session(selected_models_names, prompts, crunch_mode=False, resume_state=None):
    """
    Runs the benchmark session. 
    If crunch_mode is True, runs in parallel.
    If crunch_mode is False, runs sequentially (max_workers=1).
    """
    
    # RAM Check for Crunch Mode
    if crunch_mode and not resume_state:
        all_models_data = get_available_models_full()
        total_size_bytes = 0
        for m in all_models_data:
            if m['name'] in selected_models_names:
                total_size_bytes += m['size']
        
        total_size_gb = total_size_bytes / (1024**3)
        sys_ram_gb = get_system_memory_gb()
        limit_gb = sys_ram_gb * 0.80
        
        if total_size_gb > limit_gb:
            cont = button_dialog(
                title="RAM Warning",
                text=f"Selected models require {total_size_gb:.2f} GB RAM.\n"
                     f"Limit (80% of System): {limit_gb:.2f} GB.\n\n"
                     "System might become unresponsive.",
                buttons=[("Continue", True), ("Cancel", False)]
            ).run()
            if not cont: return

    # Session Setup
    if resume_state:
        session_id = resume_state['session_id']
        heatmap_data = resume_state['heatmap_data']
        # Ensure directories exist (in case they were deleted manually)
        create_session_directories(selected_models_names, session_id)
    else:
        session_id = get_session_id()
        create_session_directories(selected_models_names, session_id)
        heatmap_data = {} 
        for p in prompts:
            heatmap_data[p['id']] = {}
            for m in selected_models_names:
                heatmap_data[p['id']][m] = {}
        
        # Initial Save
        save_memory({
            "type": "benchmark",
            "session_id": session_id,
            "models": selected_models_names,
            "prompts": prompts, # Save prompts in case CSV changes
            "crunch_mode": crunch_mode,
            "heatmap_data": heatmap_data
        })

    # Progress Bar
    progress = Progress(
        SpinnerColumn(spinner_name="dots12", style="bold magenta"),
        TextColumn("[bold magenta]{task.description}", justify="right"),
        BarColumn(bar_width=None, style="dim white", complete_style="bold magenta", finished_style="bold green"),
        TaskProgressColumn(text_format="[progress.percentage]{task.percentage:>3.2f}%"),
        TimeElapsedColumn(),
        expand=True
    )
    
    model_tasks = {}
    total_ops_per_model = len(prompts) * len(TEMPERATURES)
    for model in selected_models_names:
        task_id = progress.add_task(f"[cyan]{model}", total=total_ops_per_model)
        model_tasks[model] = task_id
        
        # Fast forward progress if resuming
        if resume_state:
            completed_ops = 0
            for p in prompts:
                q_id = p['id']
                if q_id in heatmap_data and model in heatmap_data[q_id]:
                    completed_ops += len(heatmap_data[q_id][model])
            progress.update(task_id, completed=completed_ops)

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
                
                # Check if already done
                is_done = False
                if q_id in heatmap_data and model_name in heatmap_data[q_id]:
                    # Check float or string key
                    if temp in heatmap_data[q_id][model_name] or str(temp) in heatmap_data[q_id][model_name]:
                        is_done = True
                
                if is_done:
                    continue

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
                    
                    # Update Data
                    if q_id not in heatmap_data: heatmap_data[q_id] = {}
                    if model_name not in heatmap_data[q_id]: heatmap_data[q_id][model_name] = {}
                    heatmap_data[q_id][model_name][temp] = filename # Use float key in memory
                    
                else:
                    if q_id not in heatmap_data: heatmap_data[q_id] = {}
                    if model_name not in heatmap_data[q_id]: heatmap_data[q_id][model_name] = {}
                    heatmap_data[q_id][model_name][temp] = "ERROR"
                    play_sound("fail")
                
                # Save State
                save_memory({
                    "type": "benchmark",
                    "session_id": session_id,
                    "models": selected_models_names,
                    "prompts": prompts,
                    "crunch_mode": crunch_mode,
                    "heatmap_data": heatmap_data
                })
                
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
        cancel_event = True # Treat critical errors as cancellation to preserve memory
        console.print(f"[bold red]Critical Error:[/bold red] {e}")
        play_sound("error")

    if not cancel_event:
        play_sound("success")
        generate_heatmaps(heatmap_data, selected_models_names, session_id)
        clear_memory()
    else:
        console.print("[yellow]Session paused/cancelled. Progress saved.[/yellow]")

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
    choices = [
        ("ALL", "PURGE ALL HISTORY (All Models)"),
        ("EVALS", "Purge All Evaluation Files (*_eval.json)")
    ] + [(m, m) for m in models]
    
    selected_model = radiolist_dialog(
        title="Purge Records - Select Model",
        text="Select a model to view sessions, or purge everything:",
        values=choices
    ).run()

    if not selected_model:
        play_sound("back")
        return

    if selected_model == "EVALS":
        confirm = button_dialog(
            title="Confirm Purge Evaluations",
            text="Are you sure you want to delete ALL evaluation files (*_eval.json)?\nThis will keep the session data but remove analysis.",
            buttons=[("Yes, Delete Evals", True), ("Cancel", False)]
        ).run()
        
        if confirm:
            play_sound("delete")
            count = 0
            for root, dirs, files in os.walk(OUTPUT_DIR):
                for file in files:
                    if file.endswith("_eval.json"):
                        try:
                            os.remove(os.path.join(root, file))
                            count += 1
                        except Exception:
                            pass
            message_dialog(title="Success", text=f"Deleted {count} evaluation files.").run()
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
    try:
        selected_sessions = checkboxlist_dialog(
            title=f"Purge Sessions - {selected_model}",
            text="Select sessions to DELETE (Space to select):",
            values=[(s, s) for s in sessions]
        ).run()
    except IndexError:
        play_sound('error')
        console.print("[red]UI Error: Mouse interaction failed. Please use keyboard navigation.[/red]")
        time.sleep(2)
        return

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


def compute_accuracy_from_data(data):
    """Heuristic extraction of accuracy (0-1) from an evaluation JSON structure.
    Returns float between 0 and 1 or None if not found.
    """
    try:
        # If it's a dict with a direct accuracy field
        if isinstance(data, dict):
            for key in ("accuracy", "acc", "accuracy_pct", "accuracy_percent", "accuracy_percentile", "score"):
                if key in data:
                    val = data[key]
                    try:
                        valf = float(val)
                        # normalize percentages >1
                        if valf > 1:
                            valf = valf / 100.0
                        if 0.0 <= valf <= 1.0:
                            return valf
                    except Exception:
                        pass

            # If evaluator returned a category label like 'factual' / 'incoherent'
            for key in ("label", "category", "classification", "result", "judgment", "verdict"):
                if key in data and isinstance(data[key], str):
                    valstr = data[key].strip().lower()
                    if 'factual' in valstr or 'correct' in valstr or 'true' in valstr:
                        return 1.0
                    if 'incoherent' in valstr or 'incorrect' in valstr or 'false' in valstr or 'halluc' in valstr:
                        return 0.0

            # correct/total pattern
            if "correct" in data and "total" in data:
                try:
                    c = float(data.get("correct", 0))
                    t = float(data.get("total", 0))
                    if t > 0:
                        return max(0.0, min(1.0, c / t))
                except Exception:
                    pass

            # results list with booleans or dicts containing 'correct' or textual labels
            if "results" in data and isinstance(data["results"], list) and data["results"]:
                items = data["results"]
                truths = 0
                total = 0
                for it in items:
                    total += 1
                    if isinstance(it, bool):
                        if it:
                            truths += 1
                    elif isinstance(it, dict) and it.get("correct") in (True, "true", "True", 1, "1"):
                        truths += 1
                    elif isinstance(it, dict):
                        # check textual label
                        for labk in ("label", "category", "classification", "result"):
                            if labk in it and isinstance(it[labk], str):
                                s = it[labk].strip().lower()
                                if 'factual' in s or 'correct' in s or 'true' in s:
                                    truths += 1
                                    break
                if total:
                    return truths / total

        # If it's a list of booleans or dicts
        if isinstance(data, list) and data:
            truths = 0
            total = 0
            for it in data:
                total += 1
                if isinstance(it, bool) and it:
                    truths += 1
                elif isinstance(it, dict) and it.get("correct") in (True, "true", "True", 1, "1"):
                    truths += 1
                elif isinstance(it, dict):
                    for labk in ("label", "category", "classification", "result"):
                        if labk in it and isinstance(it[labk], str):
                            s = it[labk].strip().lower()
                            if 'factual' in s or 'correct' in s or 'true' in s:
                                truths += 1
                                break
            if total:
                return truths / total
    except Exception:
        pass
    return None


def generate_graph_from_csv(csv_path, output_dir, graph_type='line'):
    """Generates a graph from a CSV file using matplotlib."""
    if not plt:
        return

    try:
        os.makedirs(output_dir, exist_ok=True)
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        if not rows:
            return

        header = rows[0]
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.png")

        plt.figure(figsize=(12, 6))

        if graph_type == 'trends':
            # Bar chart for trends
            # Header: Model, Slope, Intercept, Trend_Direction, R_Squared
            models = []
            slopes = []
            colors = []
            
            for row in rows[1:]:
                if len(row) < 4: continue
                model = row[0]
                try:
                    slope = float(row[1])
                    direction = row[3]
                    models.append(model)
                    slopes.append(slope)
                    if direction == 'Positive': colors.append('green')
                    elif direction == 'Negative': colors.append('red')
                    else: colors.append('gray')
                except:
                    pass
            
            if models:
                plt.bar(models, slopes, color=colors)
                plt.title(f"Trend Analysis (Slope) - {base_name}")
                plt.xlabel("Model")
                plt.ylabel("Slope")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()

        else:
            # Line plot for heatmap, derivative, pct_change
            # Header: Model, T0.0, T0.1 ...
            temps = []
            # Parse temps from header
            for h in header[1:]:
                try:
                    temps.append(float(h))
                except:
                    temps.append(h) # Fallback if not float

            for row in rows[1:]:
                model = row[0]
                y_values = []
                x_values = []
                
                for i, v in enumerate(row[1:]):
                    if i < len(temps):
                        if v and v != '' and v != 'INF':
                            try:
                                val = float(v)
                                y_values.append(val)
                                x_values.append(temps[i])
                            except:
                                pass
                
                if x_values and y_values:
                    plt.plot(x_values, y_values, marker='o', label=model)

            title_map = {
                'line': 'Accuracy vs Temperature',
                'derivative': 'Derivative vs Temperature',
                'pct_change': 'Percent Change vs Temperature'
            }
            
            plt.title(f"{title_map.get(graph_type, 'Graph')} - {base_name}")
            plt.xlabel("Temperature")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

    except Exception as e:
        console.print(f"[red]Error generating graph for {csv_path}: {e}[/red]")

def aggregate_jsons_and_heatmap():
    """Scan `OUTPUT_DIR` for JSON files, separate syntactically invalid files,
    create a super JSON of valid parses, and produce a heatmap CSV of accuracy
    by model (rows) vs temperature (columns).
    Also generates category-specific heatmaps in output/specifics/.
    """
    summaries_dir = os.path.join(OUTPUT_DIR, "Summaries")
    specifics_dir = os.path.join(OUTPUT_DIR, "specifics")
    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(specifics_dir, exist_ok=True)

    # Load categories
    categories_map = load_categories()

    valid_entries = []
    invalid_entries = []

    for root, dirs, files in os.walk(OUTPUT_DIR):
        # skip summaries dir
        if os.path.basename(root) == "Summaries" or os.path.basename(root) == "specifics":
            continue
        for fname in files:
            if not fname.lower().endswith('.json'):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # infer model from path: OUTPUT_DIR/<model>/<session>/file.json
                rel = os.path.relpath(path, OUTPUT_DIR)
                parts = rel.split(os.sep)
                model = parts[0] if len(parts) >= 3 else (parts[-2] if len(parts) >= 2 else 'unknown')

                # infer temperature and ID from filename (.tXX,numID pattern)
                temp = None
                q_id = None
                
                # Regex to capture temp and ID
                # Expecting: model.t01,num5_eval.json or similar
                m = re.search(r'\.t(\d{2})', fname)
                if m:
                    try:
                        temp = int(m.group(1)) / 10.0
                    except Exception:
                        temp = None
                
                m_id = re.search(r',num(\d+)', fname)
                if m_id:
                    q_id = m_id.group(1)

                if temp is None:
                    # try common keys
                    if isinstance(data, dict):
                        temp = data.get('temperature') or data.get('temp') or data.get('temperature_setting')
                        try:
                            if temp is not None:
                                temp = float(temp)
                        except Exception:
                            temp = None

                accuracy = compute_accuracy_from_data(data)
                
                category = categories_map.get(q_id, "Uncategorized")

                valid_entries.append({
                    'file': path,
                    'model': model,
                    'temp': temp,
                    'q_id': q_id,
                    'category': category,
                    'accuracy': accuracy,
                    'data': data
                })
            except Exception as e:
                invalid_entries.append({'file': path, 'error': str(e)})

    # Save aggregated JSONs
    super_path = os.path.join(summaries_dir, 'super_valid.json')
    invalid_path = os.path.join(summaries_dir, 'invalid_files.json')
    try:
        with open(super_path, 'w', encoding='utf-8') as f:
            json.dump(valid_entries, f, indent=2)
    except Exception:
        pass
    try:
        with open(invalid_path, 'w', encoding='utf-8') as f:
            json.dump(invalid_entries, f, indent=2)
    except Exception:
        pass

    # Helper to build and write heatmap
    def write_heatmap(entries, output_path):
        heatmap = {}
        temps_list = list(TEMPERATURES) if TEMPERATURES else []
        
        for e in entries:
            model = e.get('model') or 'unknown'
            temp = e.get('temp')
            acc = e.get('accuracy')
            if acc is None or temp is None:
                continue
            try:
                t = float(temp)
            except Exception:
                continue

            # find nearest bin
            if not temps_list:
                b = t
            else:
                step = temps_list[1] - temps_list[0] if len(temps_list) > 1 else 0.1
                tol = abs(step) / 2.0 + 1e-9
                nearest = None
                mind = None
                for bin_t in temps_list:
                    d = abs(bin_t - t)
                    if mind is None or d < mind:
                        mind = d
                        nearest = bin_t
                if mind is not None and mind <= tol:
                    b = nearest
                else:
                    continue

            heatmap.setdefault(model, {}).setdefault(b, []).append(float(acc))

        models = sorted(heatmap.keys())
        
        if not models:
            return None

        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvf:
                writer = csv.writer(csvf)
                header = ['Model'] + [f"{t:.1f}" for t in temps_list]
                writer.writerow(header)
                for model in models:
                    row = [model]
                    for t in temps_list:
                        vals = heatmap.get(model, {}).get(t, [])
                        if vals:
                            avg = statistics.mean(vals)
                            avg = max(0.0, min(1.0, avg))
                            row.append(f"{avg:.4f}")
                        else:
                            row.append('')
                    writer.writerow(row)
            return temps_list
        except Exception:
            return None

    # 1. Main Heatmap
    csv_basename = f"heatmap_{get_session_id()}.csv"
    csv_path = os.path.join(summaries_dir, csv_basename)
    temps_used = write_heatmap(valid_entries, csv_path)
    
    # Generate Graph for Main Heatmap
    graphs_dir = os.path.join(OUTPUT_DIR, "graphs")
    generate_graph_from_csv(csv_path, os.path.join(graphs_dir, "main"), graph_type='line')

    csv_id = None
    if temps_used:
        try:
            csv_id, entry = _register_csv_entry(summaries_dir, csv_path, temps_used)
        except Exception:
            pass

    # 2. Category Specific Heatmaps
    # Group entries by category
    entries_by_cat = {}
    for e in valid_entries:
        cat = e.get('category', 'Uncategorized')
        if cat not in entries_by_cat:
            entries_by_cat[cat] = []
        entries_by_cat[cat].append(e)
    
    generated_specifics = []
    for cat, entries in entries_by_cat.items():
        safe_cat = "".join([c if c.isalnum() else "_" for c in cat])
        cat_csv_name = f"heatmap_{safe_cat}_{get_session_id()}.csv"
        cat_csv_path = os.path.join(specifics_dir, cat_csv_name)
        if write_heatmap(entries, cat_csv_path):
            generated_specifics.append(cat)
            # Generate Graph for Specific Heatmap
            generate_graph_from_csv(cat_csv_path, os.path.join(graphs_dir, "specifics"), graph_type='line')

    msg = f"Processed {len(valid_entries)} valid JSONs.\nHeatmap CSV: {csv_path}"
    if csv_id:
        msg += f"\nCSV ID: {csv_id}"
    if generated_specifics:
        msg += f"\nGenerated {len(generated_specifics)} category heatmaps in output/specifics/"
        
    message_dialog(title="Aggregation Complete", text=msg).run()


def generate_heatmap_derivative(csv_path=None):
    """Read `heatmap.csv` and produce:
    1. `heatmap_derivative.csv`: d/dx (derivative along temp axis).
    2. `heatmap_pct_change.csv`: Percent change vs previous point.
    3. `heatmap_trends.csv`: Trend analysis (Slope/Direction).

    Behavior:
    - If csv_path is None, presents a multi-selection UI.
    - Processes all selected CSVs.
    """
    summaries_dir = os.path.join(OUTPUT_DIR, "Summaries")
    
    # 1. Determine Input CSVs
    selected_paths = []
    if csv_path:
        selected_paths = [csv_path]
    else:
        # UI Selection
        index_path = os.path.join(summaries_dir, 'csv_index.json')
        choices = []
        index = {}
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    index = json.load(f)
            except Exception:
                index = {}

        if index:
            for cid, meta in sorted(index.items(), key=lambda x: x[1].get('created_at', ''), reverse=True):
                created = meta.get('created_at', '')
                fname = meta.get('filename', '')
                temps_preview = ','.join([f"{t:.1f}" for t in meta.get('temps', [])])
                label = f"[{cid}] {fname} ({created})\nTemps: {temps_preview}"
                choices.append((cid, label))
        else:
            for fn in sorted(os.listdir(summaries_dir), reverse=True):
                if fn.startswith('heatmap_') and fn.lower().endswith('.csv') and 'derivative' not in fn and 'pct_change' not in fn and 'trends' not in fn:
                    full = os.path.join(summaries_dir, fn)
                    ts = datetime.datetime.fromtimestamp(os.path.getmtime(full)).isoformat()
                    label = f"{fn} ({ts})"
                    choices.append((full, label))

        if not choices:
            message_dialog(title='Error', text=f'No heatmap CSVs found in: {summaries_dir}').run()
            return

        # Add Select All option
        choices.insert(0, ("ALL", " [ SELECT ALL ]"))

        # Multi-select UI
        try:
            sel = checkboxlist_dialog(
                title='Select Heatmap CSVs', 
                text='Choose heatmap CSVs to analyze (Space to select):', 
                values=choices
            ).run()
        except IndexError:
            play_sound('error')
            console.print("[red]UI Error: Mouse interaction failed. Please use keyboard navigation.[/red]")
            time.sleep(2)
            return
        
        if not sel:
            play_sound('back')
            return

        # Map selections to paths
        if "ALL" in sel:
            # Select all real choices (excluding "ALL" itself if it was in the list of values, but here we iterate choices)
            # choices is list of (value, label)
            for val, label in choices:
                if val != "ALL":
                    if val in index:
                        selected_paths.append(index[val].get('path'))
                    else:
                        selected_paths.append(val)
        else:
            for s in sel:
                if s in index:
                    selected_paths.append(index[s].get('path'))
                else:
                    selected_paths.append(s)
        
        # Remove duplicates just in case
        selected_paths = list(set(selected_paths))

    # 2. Process Each CSV
    results_msg = []
    
    for current_csv in selected_paths:
        if not os.path.exists(current_csv):
            results_msg.append(f"Skipped (not found): {current_csv}")
            continue

        # Resolve ID
        csv_id = None
        try:
            # Try to find ID in index
            abs_csv = os.path.abspath(current_csv)
            base_csv = os.path.basename(current_csv)
            if os.path.exists(index_path):
                with open(index_path, 'r', encoding='utf-8') as f:
                    _index = json.load(f)
                for cid, meta in _index.items():
                    if os.path.abspath(meta.get('path', '')) == abs_csv or meta.get('filename', '') == base_csv:
                        csv_id = cid
                        break
        except Exception:
            pass
        
        # Fallback ID from filename or timestamp
        if not csv_id:
            # try extracting from filename heatmap_ID.csv
            m = re.search(r'heatmap_([a-f0-9]+)\.csv', base_csv)
            if m:
                csv_id = m.group(1)
            else:
                csv_id = get_session_id()

        # Read Data
        try:
            with open(current_csv, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
        except Exception as e:
            results_msg.append(f"Error reading {base_csv}: {e}")
            continue

        if not rows:
            continue

        header = rows[0]
        temps = []
        for h in header[1:]:
            try:
                temps.append(float(h))
            except Exception:
                temps.append(None)

        # --- A. Standard Derivative ---
        deriv_rows = [header]
        # --- B. Percent Change ---
        pct_rows = [header]
        # --- C. Trend Analysis ---
        trend_rows = [['Model', 'Slope', 'Intercept', 'Trend_Direction', 'R_Squared']]

        for row in rows[1:]:
            if not row:
                continue
            model = row[0]
            values = []
            for v in row[1:]:
                if v is None or v == '':
                    values.append(None)
                else:
                    try:
                        values.append(float(v))
                    except Exception:
                        values.append(None)

            # 1. Derivatives & 2. Percent Change
            d_row = [model]
            p_row = [model]
            
            n = len(values)
            
            # For Trend Calculation
            valid_points_x = []
            valid_points_y = []

            for i in range(n):
                if i >= len(temps):
                    break
                
                yi = values[i]
                xi = temps[i]
                
                if yi is not None and xi is not None:
                    valid_points_x.append(xi)
                    valid_points_y.append(yi)

                # Derivative Logic
                deriv_val = None
                if yi is not None and xi is not None:
                    # find neighbors
                    j = i - 1
                    while j >= 0 and (values[j] is None or temps[j] is None): j -= 1
                    k = i + 1
                    while k < n and (values[k] is None or temps[k] is None): k += 1
                    
                    if j >= 0 and k < n:
                        if temps[k] != temps[j]: deriv_val = (values[k] - values[j]) / (temps[k] - temps[j])
                    elif k < n:
                        if temps[k] != xi: deriv_val = (values[k] - yi) / (temps[k] - xi)
                    elif j >= 0:
                        if xi != temps[j]: deriv_val = (yi - values[j]) / (xi - temps[j])
                
                d_row.append(f"{deriv_val:.6f}" if deriv_val is not None else '')

                # Percent Change Logic
                # (yi - y_prev) / y_prev
                # Skip first point (i=0) effectively as it has no prev
                pct_val = None
                if i > 0 and yi is not None:
                    # find immediate previous valid point
                    j = i - 1
                    while j >= 0 and values[j] is None: j -= 1
                    
                    if j >= 0:
                        y_prev = values[j]
                        if y_prev != 0:
                            pct_val = (yi - y_prev) / abs(y_prev)
                        elif yi == 0:
                            pct_val = 0.0 # 0 to 0 is 0 change
                        else:
                            pct_val = float('inf') # 0 to something is infinite growth
                
                p_row.append(f"{pct_val:.4%}" if pct_val is not None and pct_val != float('inf') else ('INF' if pct_val == float('inf') else ''))

            deriv_rows.append(d_row)
            pct_rows.append(p_row)

            # 3. Trend Calculation (Linear Regression)
            if len(valid_points_x) > 1:
                try:
                    # Simple Linear Regression
                    mean_x = statistics.mean(valid_points_x)
                    mean_y = statistics.mean(valid_points_y)
                    
                    numer = sum((x - mean_x) * (y - mean_y) for x, y in zip(valid_points_x, valid_points_y))
                    denom = sum((x - mean_x) ** 2 for x in valid_points_x)
                    
                    if denom != 0:
                        slope = numer / denom
                        intercept = mean_y - (slope * mean_x)
                        
                        # R-squared
                        ss_tot = sum((y - mean_y) ** 2 for y in valid_points_y)
                        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(valid_points_x, valid_points_y))
                        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
                        
                        direction = "Neutral"
                        if slope > 0.05: direction = "Positive"
                        elif slope < -0.05: direction = "Negative"
                        
                        trend_rows.append([model, f"{slope:.4f}", f"{intercept:.4f}", direction, f"{r2:.4f}"])
                    else:
                        trend_rows.append([model, "N/A", "N/A", "Undefined", "N/A"])
                except Exception:
                    trend_rows.append([model, "Error", "", "", ""])
            else:
                trend_rows.append([model, "Insufficient Data", "", "", ""])

        # Write Files
        base_name = os.path.splitext(base_csv)[0]
        graphs_dir = os.path.join(OUTPUT_DIR, "graphs")
        
        # Derivative
        out_d = os.path.join(summaries_dir, f"{base_name}_derivative.csv")
        with open(out_d, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(deriv_rows)
        generate_graph_from_csv(out_d, os.path.join(graphs_dir, "derivatives"), graph_type='derivative')
            
        # Pct Change
        out_p = os.path.join(summaries_dir, f"{base_name}_pct_change.csv")
        with open(out_p, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(pct_rows)
        generate_graph_from_csv(out_p, os.path.join(graphs_dir, "pct_change"), graph_type='pct_change')
            
        # Trends
        out_t = os.path.join(summaries_dir, f"{base_name}_trends.csv")
        with open(out_t, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(trend_rows)
        generate_graph_from_csv(out_t, os.path.join(graphs_dir, "trends"), graph_type='trends')

        results_msg.append(f"Processed {base_csv} -> Generated derivative, pct_change, trends.")

    message_dialog(title='Batch Processing Complete', text="\n".join(results_msg)).run()

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
    # Ensure only one instance runs at a time
    acquire_instance_lock()

    with console.status("[bold green]Fetching available models...[/bold green]", spinner="dots"):
        all_models_data = get_available_models_full()
    
    if not all_models_data:
        console.print("[bold red]No models found in Ollama.[/bold red]")
        sys.exit(1)

    # Main Menu Loop
    selected_models_names = []
    
    while True:
        # Check for resume state
        resume_state = load_memory()
        
        # Build a grouped top-level menu: Resume/Benchmark + Tabs
        menu_items = []
        if resume_state:
            menu_items.append(("resume", "Resume Previous Session"))
        else:
            menu_items.append(("benchmark", "Run Benchmark"))

        # Tabs
        menu_items.extend([
            ("analysis", "Analysis (Aggregate / Evaluate / Stats)"),
            ("maintenance", "Maintenance (Purge / Browse / Download)"),
            ("more", "More (Discover / GitHub)") ,
            ("exit", "Shut Down")
        ])

        action = radiolist_dialog(
            title="HEATMAP MAIN MENU",
            text="Choose an action:",
            values=menu_items
        ).run()

        if action is None or action == "exit":
            if action is None: play_sound("back")
            shutdown_animation()
            sys.exit(0)

        if action == "resume":
            if resume_state:
                r_models = resume_state.get('models', [])
                r_prompts = resume_state.get('prompts', [])
                r_crunch = resume_state.get('crunch_mode', False)

                if not r_models or not r_prompts:
                    console.print("[red]Corrupt save file. Clearing memory.[/red]")
                    clear_memory()
                    continue
                # Short intermediary screen to smooth the transition
                try:
                    from rich.panel import Panel
                    from rich.spinner import Spinner
                    with Live(Panel(Spinner('dots', text='Resuming session...'), title='Resuming', width=40), console=console, refresh_per_second=12, screen=False):
                        time.sleep(1.0)
                except Exception:
                    try:
                        play_sound('notify')
                    except Exception:
                        pass

                run_benchmark_session(r_models, r_prompts, crunch_mode=r_crunch, resume_state=resume_state)
            else:
                console.print("[red]No save state found.[/red]")
            continue

        if action == "benchmark":
            selected_models_names = select_models_ui(all_models_data)
            if selected_models_names:
                prompts = load_prompts(INPUT_CSV)
                console.print(f"[blue]Loaded {len(prompts)} prompts.[/blue]")

                crunch_mode = False
                if len(selected_models_names) > 1:
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
            continue

        # Handle Tabs
        if action == "analysis":
            analysis_items = [
                ("aggregate", "Aggregate JSONs & Create Heatmap"),
                ("heatmap_derivative", "Generate Heatmap Derivative CSV"),
                ("evaluate", "Evaluate Sessions"),
                ("stats", "View Statistics & Health"),
                ("back", "<< Back")
            ]
            sub = radiolist_dialog(title="Analysis", text="Choose analysis action:", values=analysis_items).run()
            if sub is None or sub == "back":
                play_sound("back")
                continue
            if sub == "aggregate":
                aggregate_jsons_and_heatmap()
                continue
            if sub == "heatmap_derivative":
                try:
                    generate_heatmap_derivative()
                except Exception as e:
                    message_dialog(title="Error", text=f"Failed to generate derivative CSV: {e}").run()
                continue
            if sub == "stats":
                show_stats_ui()
                continue
            if sub == "evaluate":
                evaluate_sessions_ui(all_models_data)
                continue

        if action == "maintenance":
            maint_items = [
                ("purge", "Purge Old Records"),
                ("open_output", "Open Output Folder"),
                ("inspect", "Browse & Inspect Models"),
                ("download", "Download New Models"),
                ("settings", "Settings"),
                ("back", "<< Back")
            ]
            sub = radiolist_dialog(title="Maintenance", text="Choose maintenance action:", values=maint_items).run()
            if sub is None or sub == "back":
                play_sound("back")
                continue
            if sub == "purge":
                purge_records_ui()
                continue
            if sub == "open_output":
                if not os.path.exists(OUTPUT_DIR):
                    message_dialog(title="Info", text="No output directory found.").run()
                else:
                    play_sound("notify")
                    open_output_dir(OUTPUT_DIR)
                continue
            if sub == "inspect":
                inspect_models_ui(all_models_data)
                continue
            if sub == "download":
                download_model_ui()
                with console.status("[bold green]Refreshing models...[/bold green]", spinner="dots"):
                    all_models_data = get_available_models_full()
                continue
            if sub == "settings":
                # Settings menu: change temp step/range and question range
                settings = load_settings()
                while True:
                    s_items = [
                        ("temps", f"Temperatures: {settings.get('temp_min')}..{settings.get('temp_max')} step {settings.get('temp_step')}"),
                        ("questions", f"Question Range: {settings.get('question_range', {}).get('start',1)}..{settings.get('question_range', {}).get('end','ALL')}"),
                        ("back", "<< Back")
                    ]
                    sel = radiolist_dialog(title="Settings", text="Modify settings:", values=s_items).run()
                    if sel is None or sel == "back":
                        play_sound('back')
                        break
                    if sel == 'temps':
                        # Ask for min, max, step
                        minv = input_dialog(title="Temp Min", text=f"Enter min temperature (current: {settings.get('temp_min')}):").run()
                        if minv is None:
                            continue
                        maxv = input_dialog(title="Temp Max", text=f"Enter max temperature (current: {settings.get('temp_max')}):").run()
                        if maxv is None:
                            continue
                        stepv = input_dialog(title="Temp Step", text=f"Enter temp step (current: {settings.get('temp_step')}):").run()
                        if stepv is None:
                            continue
                        try:
                            minf = float(minv)
                            maxf = float(maxv)
                            stepf = float(stepv)
                            if stepf <= 0 or maxf < minf:
                                message_dialog(title='Error', text='Invalid temperature range or step.').run()
                                continue
                            settings['temp_min'] = minf
                            settings['temp_max'] = maxf
                            settings['temp_step'] = stepf
                            save_settings(settings)
                            apply_temperature_settings(settings)
                            message_dialog(title='Saved', text=f"Temperatures updated ({minf}..{maxf} step {stepf}).").run()
                        except Exception:
                            message_dialog(title='Error', text='Invalid numeric value.').run()
                        continue
                    if sel == 'questions':
                        startv = input_dialog(title='Question Start', text=f"Enter start index (1-based, current: {settings.get('question_range', {}).get('start',1)}):").run()
                        if startv is None:
                            continue
                        endv = input_dialog(title='Question End', text=f"Enter end index or leave blank for ALL (current: {settings.get('question_range', {}).get('end','ALL')}):").run()
                        if endv is None:
                            continue
                        try:
                            s_int = int(startv)
                            e_int = None
                            if str(endv).strip() != '':
                                e_int = int(endv)
                                if e_int < s_int:
                                    message_dialog(title='Error', text='End must be >= start.').run()
                                    continue
                            settings['question_range'] = {'start': s_int, 'end': e_int}
                            save_settings(settings)
                            message_dialog(title='Saved', text=f"Question range set to {s_int}..{e_int or 'ALL'}").run()
                        except Exception:
                            message_dialog(title='Error', text='Invalid integer value.').run()
                        continue

        if action == "more":
            more_items = [
                ("get_more", "Find more models on Ollama.com"),
                ("github", "Visit GitHub Page"),
                ("back", "<< Back")
            ]
            sub = radiolist_dialog(title="More", text="Choose an action:", values=more_items).run()
            if sub is None or sub == "back":
                play_sound("back")
                continue
            if sub == "github":
                play_sound("jingle")
                webbrowser.open("https://github.com/NagusameCS/Heatmap")
                continue
            if sub == "get_more":
                play_sound("jingle")
                webbrowser.open("https://ollama.com/search")
                continue

    console.print(Panel.fit("[bold green]All tasks completed successfully![/bold green]", border_style="green"))

def load_evaluator_prompt():
    try:
        with open(EVALUATOR_PROMPT_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # Create default if missing
        default_prompt = (
            "You are an impartial evaluator. Categorize the following answer.\n"
            "Return a JSON object with 'category' and 'justification'.\n"
            "Do NOT output markdown or code blocks. Output ONLY raw JSON."
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
        SpinnerColumn(spinner_name="dots12", style="bold yellow"),
        TextColumn("[bold yellow]{task.description}", justify="right"),
        BarColumn(bar_width=None, style="dim white", complete_style="bold yellow", finished_style="bold green"),
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
            for chunk in query_ollama_stream(evaluator_model, full_prompt, 0.1, json_mode=True): # Low temp for evaluation
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
    try:
        selected_values = checkboxlist_dialog(
            title="Select Sessions to Evaluate",
            text="Pick sessions (grouped by model).\nSelect 'SELECT ALL SESSIONS' to include everything.",
            values=choices
        ).run()
    except IndexError:
        play_sound('error')
        console.print("[red]UI Error: Mouse interaction failed. Please use keyboard navigation.[/red]")
        time.sleep(2)
        return

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

def calculate_and_save_stats():
    """Calculates stats and saves to file."""
    if not os.path.exists(OUTPUT_DIR):
        return "No output directory found."

    total_size = 0
    file_count = 0
    
    # Model Runtime Stats
    # Structure: { model_name: {'total_duration': 0, 'session_count': 0} }
    model_runtimes = {}
    
    # Eval Stats
    eval_stats = {'total': 0, 'success': 0, 'failed': 0}
    
    # Walk for size and evals
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            fp = os.path.join(root, f)
            try:
                size = os.path.getsize(fp)
                total_size += size
                file_count += 1
                
                if f.endswith("_eval.json"):
                    eval_stats['total'] += 1
                    try:
                        with open(fp, 'r', encoding='utf-8') as json_file:
                            data = json.load(json_file)
                            if 'category' in data and 'justification' in data:
                                eval_stats['success'] += 1
                            else:
                                eval_stats['failed'] += 1
                    except:
                        eval_stats['failed'] += 1
            except OSError:
                pass

    # Walk for Runtimes (Session based)
    # We need to look at output/Model/SessionID
    # We can iterate top level directories in output
    for model_dir in os.listdir(OUTPUT_DIR):
        model_path = os.path.join(OUTPUT_DIR, model_dir)
        if os.path.isdir(model_path) and model_dir != "Summaries":
            
            if model_dir not in model_runtimes:
                model_runtimes[model_dir] = {'total_duration': 0, 'session_count': 0}
                
            for session_id in os.listdir(model_path):
                session_path = os.path.join(model_path, session_id)
                if os.path.isdir(session_path):
                    # Get all generation files
                    gen_files = []
                    for f in os.listdir(session_path):
                        if f.endswith(".txt") and not f.endswith("_eval.json"):
                            gen_files.append(os.path.join(session_path, f))
                    
                    if len(gen_files) > 1:
                        try:
                            # Get min and max mtime
                            mtimes = [os.path.getmtime(p) for p in gen_files]
                            start = min(mtimes)
                            end = max(mtimes)
                            duration = end - start
                            
                            # Filter out unrealistic durations (e.g. < 1s for multiple files)
                            if duration > 1:
                                model_runtimes[model_dir]['total_duration'] += duration
                                model_runtimes[model_dir]['session_count'] += 1
                        except OSError:
                            pass

    # Generate Report
    lines = []
    lines.append("HEATMAP GENERATOR STATISTICS")
    lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("="*40)
    
    # Storage
    lines.append(f"Storage Used: {format_size(total_size)}")
    lines.append(f"Total Files: {file_count}")
    
    purge_rec = "NO"
    if total_size > (1024 * 1024 * 500): # 500 MB
        purge_rec = "YES (Size > 500MB)"
    lines.append(f"Purge Recommended: {purge_rec}")
    lines.append("-" * 40)
    
    # Evals
    lines.append("Evaluation Performance")
    if eval_stats['total'] > 0:
        success_rate = (eval_stats['success'] / eval_stats['total']) * 100
        fail_rate = (eval_stats['failed'] / eval_stats['total']) * 100
        lines.append(f"Total Evaluations: {eval_stats['total']}")
        lines.append(f"Successful: {eval_stats['success']} ({success_rate:.1f}%)")
        lines.append(f"Failed/Review: {eval_stats['failed']} ({fail_rate:.1f}%)")
    else:
        lines.append("No evaluations found.")
    lines.append("-" * 40)
    
    # Runtimes
    lines.append("Average Session Runtime (Est.)")
    lines.append("(Based on file timestamps)")
    
    has_runtime = False
    for model, data in model_runtimes.items():
        if data['session_count'] > 0:
            avg_time = data['total_duration'] / data['session_count']
            lines.append(f"• {model:<20}: {avg_time:.1f}s / session")
            has_runtime = True
            
    if not has_runtime:
        lines.append("No sufficient data for runtime estimation.")
        
    report = "\n".join(lines)
    
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
        
    return report

def show_stats_ui():
    """Shows the stats page with loading animation."""
    
    # Loading Animation
    with Live(console=console, refresh_per_second=10) as live:
        # Create a layout similar to others
        progress = Progress(
            SpinnerColumn(spinner_name="dots12", style="bold cyan"),
            TextColumn("[bold cyan]Analyzing storage and logs...", justify="right"),
            expand=True
        )
        task = progress.add_task("analyze", total=None)
        
        layout = create_layout(progress, Text("Please wait...", style="dim white"))
        live.update(layout)
        
        # Simulate work / Calculate
        time.sleep(1.5) # Artificial delay for the "feel"
        report = calculate_and_save_stats()
        
    # Show Report
    message_dialog(
        title="Statistics & Health",
        text=report
    ).run()

def generate_heatmaps(heatmap_data, selected_models_names, session_id):
    """Generates CSV heatmaps from the collected data."""
    if not heatmap_data:
        return

    # Create Summaries Directory
    summary_dir = os.path.join(OUTPUT_DIR, "Summaries")
    os.makedirs(summary_dir, exist_ok=True)
    
    # 1. Main Heatmap (Links to files)
    csv_filename = f"heatmap_{session_id}.csv"
    csv_path = os.path.join(summary_dir, csv_filename)
    
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header: ID, Model1_T0.0, Model1_T0.1, ..., Model2_T0.0, ...
            header = ["ID"]
            for model in selected_models_names:
                for temp in TEMPERATURES:
                    header.append(f"{model}_T{temp}")
            writer.writerow(header)
            
            # Rows
            for q_id, models_data in heatmap_data.items():
                row = [q_id]
                for model in selected_models_names:
                    for temp in TEMPERATURES:
                        # Check float or string key
                        val = "MISSING"
                        if model in models_data:
                            if temp in models_data[model]:
                                val = models_data[model][temp]
                            elif str(temp) in models_data[model]:
                                val = models_data[model][str(temp)]
                        
                        # If it's a filename, make it a relative link or just filename
                        # For CSV readability, just filename is fine.
                        row.append(val)
                writer.writerow(row)
                
        console.print(f"[green]Heatmap CSV generated: {csv_path}[/green]")
        
        # Open the summary folder with a cross-platform helper
        try:
            open_output_dir(summary_dir)
        except Exception:
            # Best-effort fallback
            try:
                if os.name == 'nt':
                    os.startfile(summary_dir)
                elif platform.system() == 'Darwin':
                    subprocess.call(['open', summary_dir])
                else:
                    subprocess.call(['xdg-open', summary_dir])
            except Exception:
                pass

    except Exception as e:
        console.print(f"[red]Error generating heatmap CSV: {e}[/red]")

if __name__ == "__main__":
    print("Starting Heatmap Data Gatherer...")
    try:
        main()
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
        import traceback
        traceback.print_exc()

