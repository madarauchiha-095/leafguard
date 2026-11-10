$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$stateDir = Join-Path $projectRoot "app\state"
$logDir = Join-Path $stateDir "public"

if (-not (Test-Path $python)) {
    throw "Project environment not found at $python. Create the virtual environment and install requirements first."
}

$cloudflaredCommand = Get-Command cloudflared -ErrorAction SilentlyContinue
$cloudflaredPath = if ($cloudflaredCommand) { $cloudflaredCommand.Source } else { $null }
if (-not $cloudflaredPath) {
    $installedPath = Join-Path ${env:ProgramFiles(x86)} "cloudflared\cloudflared.exe"
    if (Test-Path $installedPath) { $cloudflaredPath = $installedPath }
}
if (-not $cloudflaredPath) {
    throw "cloudflared is required. Install it with: winget install --id Cloudflare.cloudflared -e"
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$tunnelOut = Join-Path $logDir "tunnel.stdout.log"
$tunnelErr = Join-Path $logDir "tunnel.stderr.log"
$serverOut = Join-Path $logDir "server.stdout.log"
$serverErr = Join-Path $logDir "server.stderr.log"
$tunnel = $null
$server = $null

try {
    Write-Output "Opening a temporary public tunnel. The link changes when the tunnel restarts."
    $tunnel = Start-Process -FilePath $cloudflaredPath `
        -ArgumentList @("tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8000") `
        -WorkingDirectory $projectRoot -PassThru `
        -RedirectStandardOutput $tunnelOut -RedirectStandardError $tunnelErr

    $publicUrl = $null
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline -and -not $publicUrl) {
        if ($tunnel.HasExited) {
            throw "cloudflared exited. Review $tunnelErr"
        }
        $tunnelLog = ""
        if (Test-Path $tunnelOut) { $tunnelLog += Get-Content $tunnelOut -Raw -ErrorAction SilentlyContinue }
        if (Test-Path $tunnelErr) { $tunnelLog += Get-Content $tunnelErr -Raw -ErrorAction SilentlyContinue }
        $match = [regex]::Match($tunnelLog, "https://[a-z0-9-]+\.trycloudflare\.com")
        if ($match.Success) {
            $publicUrl = $match.Value
        } else {
            Start-Sleep -Seconds 1
        }
    }
    if (-not $publicUrl) {
        throw "Timed out waiting for a Cloudflare Quick Tunnel URL. Review $tunnelErr"
    }

    $publicHost = ([uri]$publicUrl).Host
    $allowedHosts = @("localhost", "127.0.0.1", "testserver", $publicHost)
    if ($env:LEAFGUARD_ALLOWED_HOSTS) {
        $allowedHosts += $env:LEAFGUARD_ALLOWED_HOSTS.Split(",")
    }
    $env:LEAFGUARD_ALLOWED_HOSTS = ($allowedHosts | ForEach-Object { $_.Trim() } | Where-Object { $_ } | Select-Object -Unique) -join ","
    $env:LEAFGUARD_HTTPS = "1"

    $server = Start-Process -FilePath $python `
        -ArgumentList @("-m", "uvicorn", "app.server:app", "--host", "127.0.0.1", "--port", "8000") `
        -WorkingDirectory $projectRoot -PassThru `
        -RedirectStandardOutput $serverOut -RedirectStandardError $serverErr

    $ready = $false
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline -and -not $ready) {
        if ($server.HasExited) {
            throw "LeafGuard server exited. Review $serverErr"
        }
        try {
            $response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/healthz" -TimeoutSec 3
            $ready = $response.status -eq "ok"
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    if (-not $ready) {
        throw "LeafGuard did not become ready locally. Review $serverErr and $tunnelErr"
    }

    Write-Output ""
    Write-Output "LeafGuard is live at: $publicUrl"
    Write-Output "If this laptop's DNS does not resolve the new link immediately, try the URL from another network."
    Write-Output "Create an account at that link. Anyone with the URL can reach the site."
    Write-Output "Your laptop must stay awake and connected. Press Ctrl+C here to stop the server and tunnel."
    Write-Output ""

    while (-not $server.HasExited -and -not $tunnel.HasExited) {
        Start-Sleep -Seconds 2
    }
    if ($server.HasExited) { throw "LeafGuard server stopped. Review $serverErr" }
    throw "Cloudflare Tunnel stopped. Review $tunnelErr"
} finally {
    if ($server -and -not $server.HasExited) {
        Stop-Process -Id $server.Id -ErrorAction SilentlyContinue
    }
    if ($tunnel -and -not $tunnel.HasExited) {
        Stop-Process -Id $tunnel.Id -ErrorAction SilentlyContinue
    }
}
