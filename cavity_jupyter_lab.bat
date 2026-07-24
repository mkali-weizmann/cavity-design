@echo off
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
cd /d "%HERE%"
uv run jupyter lab --notebook-dir="%HERE%"
pause