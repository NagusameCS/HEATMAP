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
LAUNCHER = os.path.join(ROOT, 'run.py')
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


def build_entry(entry, distpath, name, clean=False):
    work = os.path.join('build', os.path.basename(distpath))
    spec = os.path.join('build', 'specs')
    # Always remove previous build artifacts to ensure a clean build
    for d in (distpath, work, spec):
        if os.path.exists(d):
            try:
                import shutil
                shutil.rmtree(d)
            except Exception:
                pass

    # If a file exists at the dist path (e.g., an executable written directly
    # into the parent folder from a previous run), remove it so we can create
    # a directory with the same name safely.
    if os.path.exists(distpath) and not os.path.isdir(distpath):
        try:
            os.remove(distpath)
        except Exception:
            pass

    os.makedirs(distpath, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    os.makedirs(spec, exist_ok=True)


    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--noconfirm',
        '--onefile',
        '--windowed',
        '--name', name,
        entry,
        '--distpath', distpath,
        '--workpath', work,
        '--specpath', spec,
    ]

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
            # Build main app into its own subfolder
            dist_main = os.path.join(ROOT, 'dist', 'mac', args.name)
            ok = build_entry(ENTRY, dist_main, args.name, clean=args.clean)
            # Create a zip archive for release upload
            try:
                import shutil
                archive_path = shutil.make_archive(os.path.join(ROOT, 'dist', 'mac', args.name), 'zip', root_dir=dist_main)
            except Exception:
                archive_path = None
            results.append(('mac', ok, dist_main, archive_path))

            # Build launcher into its own subfolder
            launcher_name = f"{args.name}Launcher"
            dist_launcher = os.path.join(ROOT, 'dist', 'mac', launcher_name)
            ok2 = build_entry(LAUNCHER, dist_launcher, launcher_name, clean=args.clean)
            try:
                import shutil
                archive_path2 = shutil.make_archive(os.path.join(ROOT, 'dist', 'mac', launcher_name), 'zip', root_dir=dist_launcher)
            except Exception:
                archive_path2 = None
            results.append(('mac-launcher', ok2, dist_launcher, archive_path2))

    if do_win:
        if platform.system() != 'Windows':
            print('Skipping Windows build: must run on Windows to build a .exe (no cross-compile).')
        else:
            dist_main = os.path.join(ROOT, 'dist', 'windows', args.name)
            ok = build_entry(ENTRY, dist_main, args.name, clean=args.clean)
            try:
                import shutil
                archive_path = shutil.make_archive(os.path.join(ROOT, 'dist', 'windows', args.name), 'zip', root_dir=dist_main)
            except Exception:
                archive_path = None
            results.append(('windows', ok, dist_main, archive_path))

            launcher_name = f"{args.name}Launcher"
            dist_launcher = os.path.join(ROOT, 'dist', 'windows', launcher_name)
            ok2 = build_entry(LAUNCHER, dist_launcher, launcher_name, clean=args.clean)
            try:
                import shutil
                archive_path2 = shutil.make_archive(os.path.join(ROOT, 'dist', 'windows', launcher_name), 'zip', root_dir=dist_launcher)
            except Exception:
                archive_path2 = None
            results.append(('windows-launcher', ok2, dist_launcher, archive_path2))

    # Update .gitignore for dist folders
    update_gitignore(['dist/', 'build/'])

    for item in results:
        if len(item) == 4:
            platform_name, ok, dist, archive = item
        else:
            platform_name, ok, dist = item
            archive = None
        msg = f"Build {platform_name}: {'OK' if ok else 'FAILED'} -> {dist}"
        if archive:
            msg += f" (archive: {archive})"
        print(msg)


if __name__ == '__main__':
    main()
