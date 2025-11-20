#!/usr/bin/env python3
"""Simple launcher to run heatmap_generator.py in a dedicated terminal window.

Usage:
  python run.py            # Opens a new terminal and runs heatmap_generator.py
  python run.py --dry-run  # Print the command that would be run (no window opened)
"""
import os
import sys
import platform
import shlex
import subprocess
import tempfile
import getpass
import psutil
from shutil import which as _shutil_which


def build_command(python_exe, script_path, extra_args=None):
    args = [shlex.quote(python_exe), shlex.quote(script_path)]
    if extra_args:
        args.extend(shlex.quote(a) for a in extra_args)
    # Prepend environment variable to mark dedicated run
    cmd = 'HEATMAP_DEDICATED=1 ' + ' '.join(args)
    # Ensure working directory preserved
    return 'cd ' + shlex.quote(os.getcwd()) + ' && ' + cmd + ' && exec $SHELL'


def open_in_new_terminal(cmd, dry_run=False):
    system = platform.system()
    if dry_run:
        print('DRY RUN:')
        print(cmd)
        return True

    try:
        if system == 'Darwin':
            # Use Terminal.app via AppleScript
            safe = cmd.replace('"', '\\"')
            osa = 'tell application "Terminal" to do script "{}"'.format(safe)
            subprocess.run(['osascript', '-e', osa])
            return True

        elif system == 'Linux':
            # Try common terminals
            if _shutil_which('gnome-terminal'):
                subprocess.Popen(['gnome-terminal', '--', 'bash', '-lc', cmd])
            elif _shutil_which('konsole'):
                subprocess.Popen(['konsole', '-e', 'bash', '-lc', cmd])
            elif _shutil_which('xterm'):
                subprocess.Popen(['xterm', '-e', cmd])
            else:
                # Fallback: run in background shell
                subprocess.Popen(['bash', '-lc', cmd])
            return True

        elif system == 'Windows':
            # Use start to open a new cmd window
            full = f'set HEATMAP_DEDICATED=1&& {cmd}'
            subprocess.Popen(['cmd', '/c', 'start', 'cmd', '/k', full])
            return True

        else:
            subprocess.Popen(cmd, shell=True)
            return True

    except Exception as e:
        print('Failed to open new terminal:', e)
        return False


def shutil_which(name):
    try:
        from shutil import which
        return which(name)
    except Exception:
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Launcher for HEATMAP in a dedicated terminal window')
    parser.add_argument('--dry-run', action='store_true', help='Print the command instead of opening a new terminal')
    parser.add_argument('--python', help='Python executable to use (defaults to current interpreter)')
    parser.add_argument('extra', nargs='*', help='Extra arguments forwarded to heatmap_generator.py')
    args = parser.parse_args()

    python_exe = args.python or sys.executable or 'python3'
    script = os.path.join(os.path.dirname(__file__), 'heatmap_generator.py')
    extra = args.extra or []

    cmd = build_command(python_exe, script, extra)
    # Single-instance check: read pidfile used by heatmap_generator
    pid_file = os.path.join(tempfile.gettempdir(), f'heatmap_generator_{getpass.getuser()}.pid')
    if os.path.exists(pid_file):
        try:
            with open(pid_file, 'r', encoding='utf-8') as f:
                existing = f.read().strip()
            if existing:
                pid = int(existing)
                if psutil.pid_exists(pid):
                    # Prompt user for action
                    print(f'Found running HEATMAP instance (PID {pid}). Choose action:')
                    print('[K]ill the other instance and start new')
                    print('[O]pen/bring the existing terminal to front')
                    print('[C]ancel launch')
                    choice = input('Choice [K/O/C]: ').strip().lower()
                    if choice == 'k':
                        print('Attempting to terminate existing process...')
                        try:
                            import signal as _signal
                            os.kill(pid, _signal.SIGTERM)
                        except Exception:
                            pass
                        # wait briefly
                        import time as _time
                        _time.sleep(1)
                        if psutil.pid_exists(pid):
                            try:
                                os.kill(pid, _signal.SIGKILL)
                            except Exception:
                                pass
                        # remove pidfile if still present
                        try:
                            if os.path.exists(pid_file):
                                os.remove(pid_file)
                        except Exception:
                            pass
                        print('Terminated previous instance (if running). Proceeding to open a new window.')
                    elif choice == 'o':
                        system = platform.system()
                        if system == 'Darwin':
                            # Activate Terminal.app (best-effort)
                            try:
                                subprocess.run(['osascript', '-e', 'tell application "Terminal" to activate'])
                                print('Brought Terminal to front. Please locate the tab running the existing HEATMAP instance.')
                            except Exception:
                                print('Could not bring Terminal to front. Please switch to the terminal running the instance (PID', pid, ').')
                        elif system == 'Linux':
                            # Try wmctrl to focus window by PID
                            winid = None
                            if _shutil_which('wmctrl'):
                                try:
                                    out = subprocess.check_output(['wmctrl', '-lp']).decode('utf-8', errors='ignore')
                                    for line in out.splitlines():
                                        parts = line.split()
                                        if len(parts) >= 3:
                                            wid = parts[0]
                                            wpid = parts[2]
                                            try:
                                                if int(wpid) == pid:
                                                    winid = wid
                                                    break
                                            except Exception:
                                                continue
                                    if winid:
                                        subprocess.run(['wmctrl', '-ia', winid])
                                        print('Focused existing terminal window.')
                                        return
                                except Exception:
                                    pass
                            print('Could not automatically focus the terminal. Please switch to PID', pid)
                        elif system == 'Windows':
                            print('Please switch to the existing HEATMAP window (PID', pid, ').')
                        else:
                            print('Please switch to the existing HEATMAP window (PID', pid, ').')
                        return
                    else:
                        print('Cancelled launch.')
                        return
        except Exception:
            pass

    ok = open_in_new_terminal(cmd, dry_run=args.dry_run)
    if not ok:
        print('Could not open a dedicated terminal window.')


if __name__ == '__main__':
    main()
