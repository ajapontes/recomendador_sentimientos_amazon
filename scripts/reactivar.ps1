<#
.SYNOPSIS
  Reactiva el entorno del proyecto, verifica CUDA/Transformers y opcionalmente corre tests y levanta la API.

.USAGE
  # Desde la raíz del repo (recomendado)
  powershell -ExecutionPolicy Bypass -File .\scripts\reactivar.ps1

  # Con smoke de Transformers y tests
  powershell -ExecutionPolicy Bypass -File .\scripts\reactivar.ps1 -RunSmoke -RunPyTests

  # Lanzar la API al final (en nueva ventana)
  powershell -ExecutionPolicy Bypass -File .\scripts\reactivar.ps1 -StartAPI -ApiPort 8002

.PARAMS
  -ProjectRoot: Ruta al proyecto. Si no se pasa, usa:
                1) $PSScriptRoot (si está disponible)
                2) C:\Proyectos\recomendador_sentimientos_amazon (fallback)
                3) El directorio actual (último recurso)
  -RunSmoke   : Ejecuta python -m src.models.test_sentiment
  -RunPyTests : Ejecuta pytest -q
  -StartAPI   : Lanza uvicorn src.api.main:app --reload --port <ApiPort> en una nueva ventana
  -ApiPort    : Puerto para la API (default 8002)
#>

[CmdletBinding()]
param(
  [string]$ProjectRoot,
  [switch]$RunSmoke = $false,
  [switch]$RunPyTests = $false,
  [switch]$StartAPI = $false,
  [int]$ApiPort = 8002
)

$ErrorActionPreference = "Stop"

function Write-Section($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-OK($msg)      { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn($msg)    { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Die($msg)           { Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

# --- 0) Resolver ProjectRoot de forma robusta ---
if (-not $ProjectRoot -or $ProjectRoot.Trim() -eq "") {
  if ($PSBoundParameters.ContainsKey('ProjectRoot')) {
    # ya viene vacío explícitamente -> ignorar
    $null = $null
  } elseif ($PSScriptRoot -and (Test-Path $PSScriptRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
  } elseif (Test-Path "C:\Proyectos\recomendador_sentimientos_amazon") {
    $ProjectRoot = "C:\Proyectos\recomendador_sentimientos_amazon"
  } else {
    $ProjectRoot = (Get-Location).Path
  }
}

if (-not (Test-Path $ProjectRoot)) { Die "No se encontró ProjectRoot: '$ProjectRoot'." }

Set-Location $ProjectRoot
Write-Section "Proyecto: $ProjectRoot"

# --- 1) Activar venv existente ---
$activatePs1 = Join-Path $ProjectRoot "env\Scripts\Activate.ps1"
$activateExe = Join-Path $ProjectRoot "env\Scripts\Activate" # fallback raro
if (Test-Path $activatePs1) {
  . $activatePs1
} elseif (Test-Path $activateExe) {
  . $activateExe
} else {
  Die "No se encontró el venv en '.\env'. Crea uno (py -3.11 -m venv env) e instala dependencias primero."
}
Write-OK "Entorno activado"

# --- 2) Ajustes de sesión (opcional) ---
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# --- 3) Verificación Python en venv ---
Write-Section "Verificación de Python"
python -V
try {
  $pyCmd = Get-Command python -ErrorAction Stop
  Write-Host "Python path: $($pyCmd.Source)"
} catch {
  Write-Warn "No se pudo resolver 'python' con Get-Command. Asegura que el venv esté activo."
}

# --- 4) Verificar settings ---
Write-Section "Verificación de settings.yaml"
try {
  python .\test_settings.py | Write-Host
  Write-OK "Settings leídos"
} catch {
  Write-Warn "Fallo al leer settings: $($_.Exception.Message)"
}

# --- 5) Verificar PyTorch + CUDA (sin heredoc; usando archivo temporal) ---
Write-Section "PyTorch + CUDA"
$code = @'
import torch, sys
print("Python:", sys.executable)
print("Torch:", torch.__version__)
print("CUDA toolkit:", torch.version.cuda)
print("CUDA disponible:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
'@
$tmp = New-TemporaryFile
Set-Content -Path $tmp -Value $code -Encoding utf8
try {
  python $tmp
} finally {
  Remove-Item $tmp -Force -ErrorAction SilentlyContinue
}

# --- 6) Smoke Transformers + CUDA (opcional) ---
if ($RunSmoke) {
  Write-Section "Smoke Transformers (DistilBERT + CUDA)"
  try {
    python -m src.models.test_sentiment
  } catch {
    Write-Warn "Smoke falló: $($_.Exception.Message)"
  }
}

# --- 7) Test suite (opcional) ---
if ($RunPyTests) {
  Write-Section "PyTest"
  try {
    pytest -q
  } catch {
    Write-Warn "PyTest reportó errores: $($_.Exception.Message)"
  }
}

# --- 8) Lanzar API (opcional, nueva ventana para no bloquear) ---
if ($StartAPI) {
  Write-Section "Iniciando API en puerto $ApiPort"
  $cmd = "uvicorn src.api.main:app --reload --port $ApiPort"
  $activateForChild = Join-Path $ProjectRoot "env\Scripts\Activate.ps1"
  if (-not (Test-Path $activateForChild)) {
    Die "No se encontró $activateForChild para la ventana de la API."
  }
  Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd `"$ProjectRoot`"; . `"$activateForChild`"; $cmd"
  Write-OK "API lanzada. Health: http://127.0.0.1:$ApiPort/health"
} else {
  Write-Warn "API no iniciada (usa -StartAPI para lanzarla)."
}

Write-OK "Reactivación y verificaciones completadas."
