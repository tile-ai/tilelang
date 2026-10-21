$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$cudaVersion = "13.4"
$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$cudaVersion"
$installerUrl = "https://packages.nvidia.com/prerelease/cuda/13.4.0/local_installers/cuda_13.4.0_windows_arm64.exe"
$installerSha256 = "a1f68c81160b16d519c4087788b9c07de41306c3f1b872471ceee0996621374d"
$installerDir = Join-Path $env:RUNNER_TEMP "cuda-installer"
$installer = Join-Path $installerDir "cuda_13.4.0_windows_arm64.exe"
$partialInstaller = "$installer.part"
$logDir = Join-Path $env:RUNNER_TEMP "cuda-install-logs"

New-Item -ItemType Directory -Force -Path $installerDir, $logDir | Out-Null

Write-Host "Downloading CUDA $cudaVersion for Windows ARM64"
curl.exe `
  --fail `
  --location `
  --retry 5 `
  --retry-all-errors `
  --retry-delay 10 `
  --continue-at - `
  --output $partialInstaller `
  $installerUrl
if ($LASTEXITCODE -ne 0) {
  throw "CUDA installer download failed with curl exit code $LASTEXITCODE"
}
Move-Item -Path $partialInstaller -Destination $installer -Force

$actualSha256 = (Get-FileHash -Algorithm SHA256 -Path $installer).Hash.ToLowerInvariant()
if ($actualSha256 -ne $installerSha256) {
  Remove-Item -Path $installer -Force
  throw "CUDA installer SHA-256 mismatch: expected $installerSha256, got $actualSha256"
}

$components = @(
  "crt_13.4",
  "cudart_13.4",
  "curand_13.4",
  "curand_dev_13.4",
  "nvcc_13.4",
  "nvjitlink_13.4",
  "nvrtc_13.4",
  "nvrtc_dev_13.4",
  "nvvm_13.4",
  "thrust_13.4"
)
$arguments = @("-s", "-n", "-log:$logDir", "-loglevel:6") + $components

Write-Host "Installing CUDA components: $($components -join ', ')"
$process = Start-Process -FilePath $installer -ArgumentList $arguments -Wait -PassThru
Remove-Item -Path $installer -Force

if ($process.ExitCode -notin @(0, 3010)) {
  Get-ChildItem -Path $logDir -File -Recurse |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 3 |
    ForEach-Object {
      Write-Host "CUDA installer log: $($_.FullName)"
      Get-Content -Path $_.FullName -Tail 120
    }
  throw "CUDA installer failed with exit code $($process.ExitCode)"
}

$nvcc = Join-Path $cudaRoot "bin\nvcc.exe"
$cicc = Join-Path $cudaRoot "nvvm\bin\cicc.exe"
if (!(Test-Path $nvcc) -or !(Test-Path $cicc)) {
  throw "CUDA installation is incomplete under $cudaRoot"
}

"CUDA_PATH=$cudaRoot" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
"CUDA_HOME=$cudaRoot" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
"CUDAToolkit_ROOT=$cudaRoot" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
"CUDA_PATH_V13_4=$cudaRoot" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
"$cudaRoot\bin" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append

$env:CUDA_PATH = $cudaRoot
$env:CUDA_HOME = $cudaRoot
$env:CUDAToolkit_ROOT = $cudaRoot
$env:Path = "$cudaRoot\bin;$env:Path"
& $nvcc --version
