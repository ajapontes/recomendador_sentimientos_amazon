<#
.SYNOPSIS
    Desactiva el entorno virtual del proyecto y opcionalmente detiene la API.

.DESCRIPTION
    - Si el entorno virtual está activo, lo desactiva.
    - Elimina variables de entorno temporales (por ejemplo las de HuggingFace).
    - Si se indica -KillAPI, busca procesos uvicorn (API FastAPI) y los detiene.

.PARAMETER KillAPI
    Indica si se deben detener procesos uvicorn que estén corriendo en el puerto
    especificado en -ApiPort.

.PARAMETER ApiPort
    Puerto en el que la API está escuchando (por defecto 8002).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\desactivar.ps1
    # Solo desactiva el entorno virtual.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\desactivar.ps1 -KillAPI
    # Desactiva el entorno virtual y detiene la API en el puerto por defecto (8002).
#>

[CmdletBinding()]
param(
    [switch]$KillAPI,
    [int]$ApiPort = 8002
)

Write-Host "=== Desactivando entorno: Recomendador Sentimientos Amazon ===" -ForegroundColor Cyan

# 1) Desactivar entorno virtual si está activo
if (Test-Path Env:\VIRTUAL_ENV) {
    Write-Host "-> Desactivando venv actual: $env:VIRTUAL_ENV" -ForegroundColor Yellow
    deactivate
} else {
    Write-Host "-> No se detectó un venv activo en esta sesión." -ForegroundColor DarkYellow
}

# 2) Limpiar variables temporales
if ($env:HF_HUB_DISABLE_SYMLINKS_WARNING) {
    Write-Host "-> Limpiando variable HF_HUB_DISABLE_SYMLINKS_WARNING" -ForegroundColor Yellow
    Remove-Item Env:\HF_HUB_DISABLE_SYMLINKS_WARNING -ErrorAction SilentlyContinue
}

# 3) (Opcional) Detener procesos de la API
if ($KillAPI) {
    Write-Host "-> Buscando procesos uvicorn en el puerto $ApiPort ..." -ForegroundColor Yellow
    $pids = netstat -ano | Select-String ":$ApiPort\s" | ForEach-Object {
        ($_ -split '\s+')[-1]
    } | Sort-Object -Unique
    if ($pids) {
        foreach ($pid in $pids) {
            try {
                Stop-Process -Id $pid -Force -ErrorAction Stop
                Write-Host "   Proceso API detenido (PID $pid)" -ForegroundColor Green
            } catch {
                Write-Host "   No se pudo detener PID $pid : $_" -ForegroundColor Red
            }
        }
    } else {
        Write-Host "-> No se encontraron procesos uvicorn en el puerto $ApiPort." -ForegroundColor DarkYellow
    }
}

Write-Host "=== Entorno desactivado. Puedes cerrar esta ventana con seguridad. ===" -ForegroundColor Cyan

#Solo desactivar el entorno:
#powershell -ExecutionPolicy Bypass -File .\scripts\desactivar.ps1
#Desactivar y detener la API (por defecto puerto 8002):
#powershell -ExecutionPolicy Bypass -File .\scripts\desactivar.ps1 -KillAPI
