@echo off
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
cd /d "%HERE%"
rem Clear a stale VIRTUAL_ENV so it can't override the environment uv picks.
set "VIRTUAL_ENV="
rem Keep the uv environment off the Dropbox-synced tree (avoids file-lock errors
rem during sync). Respects an already-set value; otherwise defaults to C:\venvs.
if not defined UV_PROJECT_ENVIRONMENT set "UV_PROJECT_ENVIRONMENT=C:\venvs\cavity-design"
uv run jupyter lab --notebook-dir="%HERE%"
pause