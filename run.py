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
from prompt_toolkit.shortcuts import radiolist_dialog, message_dialog
from prompt_toolkit.styles import Style


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

    style = Style.from_dict({'dialog': 'bg:#88ff88'})

    def check_running_instance():
        pid_file = os.path.join(tempfile.gettempdir(), f'heatmap_generator_{getpass.getuser()}.pid')
        if os.path.exists(pid_file):
            try:
                with open(pid_file, 'r', encoding='utf-8') as f:
                    existing = f.read().strip()
                if existing and psutil.pid_exists(int(existing)):
                    return int(existing)
            except Exception:
                return None
        return None

    def kill_instance(pid):
        try:
            import signal as _signal
            os.kill(pid, _signal.SIGTERM)
        except Exception:
            pass
        import time as _time
        _time.sleep(1)
        if psutil.pid_exists(pid):
            try:
                os.kill(pid, _signal.SIGKILL)
            except Exception:
                pass
        try:
            pid_file = os.path.join(tempfile.gettempdir(), f'heatmap_generator_{getpass.getuser()}.pid')
            if os.path.exists(pid_file):
                os.remove(pid_file)
        except Exception:
            pass

    def bring_existing_to_front(pid):
        system = platform.system()
        if system == 'Darwin':
            try:
                subprocess.run(['osascript', '-e', 'tell application "Terminal" to activate'])
                return True
            except Exception:
                return False
        elif system == 'Linux':
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
                                    subprocess.run(['wmctrl', '-ia', wid])
                                    return True
                            except Exception:
                                continue
                except Exception:
                    return False
            return False
        else:
            return False

    def launch_dedicated():
        cmd = build_command(python_exe, script, extra)
        ok = open_in_new_terminal(cmd, dry_run=args.dry_run)
        if not ok:
            message_dialog(title='Error', text='Could not open a dedicated terminal window.').run()

    def launch_current():
        cmd_list = [python_exe, script] + list(extra)
        try:
            subprocess.run(cmd_list)
        except Exception as e:
            message_dialog(title='Error', text=f'Failed to run: {e}').run()

    while True:
        running_pid = check_running_instance()
        status = f'Running PID: {running_pid}' if running_pid else 'No running instance detected.'
        choices = [
            ('dedicated', 'Open in Dedicated Terminal'),
            ('current', 'Run in Current Terminal'),
            ('status', f'Status ({status})'),
            ('kill', 'Force Kill Existing Instance'),
            ('quit', 'Quit')
        ]

        action = radiolist_dialog(title='HEATMAP Launcher', text='Choose an action:', values=choices, style=style).run()

        if not action or action == 'quit':
            break

        if action == 'dedicated':
            if running_pid:
                confirm = radiolist_dialog(title='Instance Exists', text=f'Instance {running_pid} detected. What to do?', values=[('abort','Abort launch'),('kill','Kill and launch'),('bring','Bring existing to front')]).run()
                if confirm == 'abort' or confirm is None:
                    continue
                if confirm == 'kill':
                    kill_instance(running_pid)
                    launch_dedicated()
                elif confirm == 'bring':
                    ok = bring_existing_to_front(running_pid)
                    if not ok:
                        message_dialog(title='Info', text='Could not focus existing window.').run()
            else:
                launch_dedicated()

        elif action == 'current':
            if running_pid:
                proceed = radiolist_dialog(title='Proceed?', text=f'Instance {running_pid} detected. Proceed to run here?', values=[('yes','Yes'),('no','No')]).run()
                if proceed != 'yes':
                    continue
            launch_current()

        elif action == 'status':
            if running_pid:
                message_dialog(title='Status', text=f'HEATMAP running (PID {running_pid})').run()
            else:
                message_dialog(title='Status', text='No running HEATMAP instance found.').run()

        elif action == 'kill':
            if running_pid:
                confirm = radiolist_dialog(title='Confirm Kill', text=f'Kill HEATMAP PID {running_pid}?', values=[('y','Yes'),('n','No')]).run()
                if confirm == 'y':
                    kill_instance(running_pid)
                    message_dialog(title='Killed', text='Process terminated (if running).').run()
            else:
                message_dialog(title='Info', text='No running instance to kill.').run()


if __name__ == '__main__':
    main()
