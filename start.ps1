# GridLock Intelligence Platform — Demo Launcher
$key = [System.Environment]::GetEnvironmentVariable("GROQ_API_KEY", "User")
if (-not $key) {
    Write-Host "WARNING: GROQ_API_KEY not set. AI Assistant will require manual key entry." -ForegroundColor Yellow
} else {
    $env:GROQ_API_KEY = $key
    Write-Host "Groq API key loaded." -ForegroundColor Green
}
Write-Host "Starting GridLock on http://localhost:8502 ..." -ForegroundColor Cyan
python -m streamlit run "$PSScriptRoot\app.py" --server.port 8502
