@echo off
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
cd /d "%HERE%"
rem Clear a stale VIRTUAL_ENV (e.g. the old C:\venvs\cavity-design) so uv uses .venv
set "VIRTUAL_ENV="
uv run jupyter lab --notebook-dir="%HERE%"
pause