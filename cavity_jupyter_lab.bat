@echo off
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
cd /d "%HERE%"
"C:\venvs\cavity-design\Scripts\jupyter.exe" lab --notebook-dir="%HERE%"
pause