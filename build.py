#!/usr/bin/env python3
"""Build helper using PyInstaller to create macOS .app and Windows .exe bundles.

This script runs PyInstaller with sensible defaults and places artifacts in:
- `dist/mac/`     -> macOS .app bundle (when run on macOS)
- `dist/windows/` -> Windows .exe (when run on Windows)

Note: PyInstaller cannot cross-compile between OSes reliably. Run this script on the
target platform you want to build for. The script will detect the OS and skip builds
that aren't supported on the current system.

Usage:
  python build.py --all         # build for current platform(s) as appropriate
  python build.py --mac         # attempt mac build (only on macOS)
  python build.py --win         # attempt windows build (only on Windows)
  python build.py --name MyApp  # override application name
  python build.py --clean       # remove PyInstaller build artifacts before building
"""
from __future__ import annotations
import os
import sys
import platform
import subprocess
import argparse

ROOT = os.path.abspath(os.path.dirname(__file__))
ENTRY = os.path.join(ROOT, 'heatmap_generator.py')
DEFAULT_NAME = 'Heatmap'


def run_cmd(cmd, env=None):
    print('Running:', ' '.join(cmd))
    try:
        subprocess.check_call(cmd, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print('Command failed:', e)
        return False


def ensure_pyinstaller():
    try:
        import PyInstaller  # type: ignore
        return True
    except Exception:
        print('\nPyInstaller is not installed in this environment.\n')
        print('Install with: pip install pyinstaller')
        return False


def build(distpath, name, clean=False, onefile=True):
    """Run PyInstaller to build. Set onefile=False for onedir builds (recommended for .app)."""
    work = os.path.join('build', os.path.basename(distpath))
    spec = os.path.join('build', 'specs')
    if clean:
        # remove previous build artifacts
        for d in (distpath, work, spec):
            if os.path.exists(d):
                try:
                    import shutil
                    shutil.rmtree(d)
                except Exception:
                    pass

    os.makedirs(distpath, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    os.makedirs(spec, exist_ok=True)

    cmd = [sys.executable, '-m', 'PyInstaller', '--noconfirm']
    # windowed makes a GUI app (no terminal window)
    cmd += ['--windowed', '--name', name, ENTRY, '--distpath', distpath, '--workpath', work, '--specpath', spec]
    if onefile:
        cmd.insert(3, '--onefile')
    else:
        # onedir mode (default when not specifying --onefile)
        pass

    return run_cmd(cmd)


def update_gitignore(entries: list[str]):
    gitignore = os.path.join(ROOT, '.gitignore')
    existing = ''
    if os.path.exists(gitignore):
        with open(gitignore, 'r', encoding='utf-8') as f:
            existing = f.read()

    with open(gitignore, 'a', encoding='utf-8') as f:
        for e in entries:
            if e not in existing:
                f.write('\n' + e + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Build for current platform')
    parser.add_argument('--mac', action='store_true', help='Attempt to build macOS .app (only on macOS)')
    parser.add_argument('--win', action='store_true', help='Attempt to build Windows .exe (only on Windows)')
    parser.add_argument('--name', default=DEFAULT_NAME, help='Application name')
    parser.add_argument('--clean', action='store_true', help='Clean previous build artifacts')

    args = parser.parse_args()

    if not ensure_pyinstaller():
        sys.exit(1)

    do_mac = args.mac
    do_win = args.win
    if args.all or (not do_mac and not do_win):
        # default: build for current host platform
        host = platform.system()
        if host == 'Darwin':
            do_mac = True
        elif host == 'Windows':
            do_win = True
        else:
            # try mac and/or win only if explicitly requested; otherwise build nothing
            print('Non-macOS/non-Windows host; use --mac/--win to target specific platforms (cross-build not supported).')
            sys.exit(0)

    results = []
    if do_mac:
        if platform.system() != 'Darwin':
            print('Skipping mac build: must run on macOS to build a .app bundle.')
        else:
            dist = os.path.join(ROOT, 'dist', 'mac')
            # For macOS .app bundles, use onedir mode (onefile + .app is deprecated/unsupported)
            ok = build(dist, args.name, clean=args.clean, onefile=False)
            results.append(('mac', ok, dist))

    if do_win:
        if platform.system() != 'Windows':
            print('Skipping Windows build: must run on Windows to build a .exe (no cross-compile).')
        else:
            dist = os.path.join(ROOT, 'dist', 'windows')
            # For Windows, onefile exe is usually desired
            ok = build(dist, args.name, clean=args.clean, onefile=True)
            results.append(('windows', ok, dist))

    # Update .gitignore for dist folders
    update_gitignore(['dist/mac/', 'dist/windows/', 'build/'])

    for platform_name, ok, dist in results:
        print(f'Build {platform_name}:', 'OK' if ok else 'FAILED', '->', dist)


if __name__ == '__main__':
    main()
