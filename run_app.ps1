$env:PYTHONUTF8 = "1"

$ErrorActionPreference = "Stop"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "        Starting NBLM Small / RAG Hub        " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

$pythonCandidates = @(
    ".\.venv\Scripts\python.exe",
    ".\venv\Scripts\python.exe",
    ".\venv_win\Scripts\python.exe",
    "python"
)

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
    if ($candidate -eq "python") {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd) {
            $pythonExe = "python"
            break
        }
    } elseif (Test-Path $candidate) {
        $pythonExe = $candidate
        break
    }
}

if (-not $pythonExe) {
    Write-Host "[Error] Python not found. Install Python 3.10+ and create a virtual environment first." -ForegroundColor Red
    exit 1
}

Write-Host "[Info] Using Python: $pythonExe" -ForegroundColor Green

$grpcPort = 50051
$grpcConn = Get-NetTCPConnection -LocalPort $grpcPort -ErrorAction SilentlyContinue | Where-Object { $_.OwningProcess -ne 0 }

if ($grpcConn) {
    Write-Host "[Info] gRPC Vision Server already running on port $grpcPort." -ForegroundColor Yellow
} else {
    Write-Host "[Info] Starting gRPC Vision Server on port $grpcPort..." -ForegroundColor Blue
    Start-Process -FilePath $pythonExe -ArgumentList "app/workers/vision_grpc_server.py" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

$webPort = 8000
$webConn = Get-NetTCPConnection -LocalPort $webPort -ErrorAction SilentlyContinue | Where-Object { $_.OwningProcess -ne 0 }
if ($webConn) {
    $procId = $webConn[0].OwningProcess
    Write-Host "[Warning] Port $webPort is already in use by PID $procId. Stopping it..." -ForegroundColor Yellow
    Stop-Process -Id $procId -Force
    Start-Sleep -Seconds 1
}

Write-Host "[Info] Starting FastAPI at http://localhost:8000" -ForegroundColor Green
Start-Process "http://localhost:8000"

& $pythonExe -m uvicorn app.main:app --port 8000
