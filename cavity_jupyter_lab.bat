@echo off
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
cd /d "%HERE%"
rem Clear a stale VIRTUAL_ENV so it cannot override the environment uv picks.
set "VIRTUAL_ENV="
rem Pin the uv environment explicitly. It must stay off the Dropbox-synced tree
rem (uv's default .venv lives inside the project and hits file-lock errors during
rem sync), and it must NOT inherit a user-level UV_PROJECT_ENVIRONMENT, because
rem `uv run` syncs whatever environment it is pointed at to match uv.lock.
set "UV_PROJECT_ENVIRONMENT=C:\venvs\cavity-design-uv2"
rem `python -m jupyterlab`, not `jupyter lab`: the jupyter.exe console-script
rem shim has vanished from the env before (AV quarantine), and the module form
rem does not depend on it.
uv run python -m jupyterlab --notebook-dir="%HERE%"
pause
