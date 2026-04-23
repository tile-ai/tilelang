@echo off
setlocal
setlocal EnableDelayedExpansion

set "NO_NVCC="
set "SYNC_ARGS="
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--no-nvcc" (
  set "NO_NVCC=1"
) else (
  set "SYNC_ARGS=!SYNC_ARGS! %~1"
)
shift
goto parse_args
:args_done

if defined VSDEVCMD_BAT if exist "%VSDEVCMD_BAT%" (
  set "VSDEVCMD=%VSDEVCMD_BAT%"
) else (
  set "VSDEVCMD="
)

if not defined VSDEVCMD (
  if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
    for /f "usebackq delims=" %%I in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find Common7\Tools\VsDevCmd.bat`) do (
      if not defined VSDEVCMD set "VSDEVCMD=%%~fI"
    )
  )
)

if not defined VSDEVCMD (
  if exist "%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe" (
    for /f "usebackq delims=" %%I in (`"%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find Common7\Tools\VsDevCmd.bat`) do (
      if not defined VSDEVCMD set "VSDEVCMD=%%~fI"
    )
  )
)

if not defined VSDEVCMD (
  echo Could not find VsDevCmd.bat. Install Visual Studio Build Tools or set VSDEVCMD_BAT. 1>&2
  exit /b 1
)

call "%VSDEVCMD%" -no_logo -arch=x64 -host_arch=x64 >nul || exit /b 1

where uv >nul 2>nul
if errorlevel 1 (
  echo Could not find 'uv' in PATH. Install uv first: https://docs.astral.sh/uv/ 1>&2
  exit /b 1
)

echo Using Visual Studio environment from: %VSDEVCMD%
if defined NO_NVCC (
  echo Running: uv sync !SYNC_ARGS!
  call uv sync !SYNC_ARGS!
) else (
  echo Running: uv sync --extra nvcc !SYNC_ARGS!
  call uv sync --extra nvcc !SYNC_ARGS!
)
exit /b %errorlevel%
