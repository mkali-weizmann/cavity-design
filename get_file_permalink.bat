@echo off
setlocal enabledelayedexpansion

set /p FILEPATH=Enter local file path: 

rem Remove surrounding quotes if pasted path has them
set "FILEPATH=%FILEPATH:"=%"

rem Get absolute path
for %%I in ("%FILEPATH%") do set "ABSFILE=%%~fI"

if not exist "%ABSFILE%" (
    echo File not found: "%ABSFILE%"
    pause
    exit /b 1
)

rem Find repo root for that file
for /f "delims=" %%R in ('git -C "%~dp1." rev-parse --show-toplevel 2^>nul') do set "REPOROOT=%%R"

rem Better: run git from the file's directory
for %%I in ("%ABSFILE%") do set "FILEDIR=%%~dpI"

for /f "delims=" %%R in ('git -C "%FILEDIR%" rev-parse --show-toplevel 2^>nul') do set "REPOROOT=%%R"

if not defined REPOROOT (
    echo Could not find a Git repository for this file.
    pause
    exit /b 1
)

rem Get relative path from repo root
for /f "delims=" %%P in ('git -C "%REPOROOT%" ls-files --full-name -- "%ABSFILE%"') do set "RELFILE=%%P"

if not defined RELFILE (
    echo File is not tracked by Git:
    echo "%ABSFILE%"
    pause
    exit /b 1
)

rem Get origin URL and commit SHA
for /f "delims=" %%U in ('git -C "%REPOROOT%" remote get-url origin') do set "REPOURL=%%U"
for /f "delims=" %%C in ('git -C "%REPOROOT%" rev-parse HEAD') do set "COMMIT=%%C"

rem Convert GitHub SSH URL to HTTPS
set "BASE=%REPOURL%"
set "BASE=%BASE:git@github.com:=https://github.com/%"

rem Remove .git suffix
if "%BASE:~-4%"==".git" set "BASE=%BASE:~0,-4%"

set "PERMALINK=%BASE%/blob/%COMMIT%/%RELFILE%"

echo.
echo %PERMALINK%
echo %PERMALINK% | clip

echo.
echo Permalink copied to clipboard.
pause