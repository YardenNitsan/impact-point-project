param(
    [string]$action
)

$devDir = ".dev"
$pidFile = "$devDir/dev_pids.json"

if (!(Test-Path $devDir)) {
    New-Item -ItemType Directory -Path $devDir | Out-Null
}

function Wait-ForPort {
    param(
        [string]$hostname,
        [int]$port
    )

    Write-Host "Waiting for ${hostname}:${port}..."

    $timeout = 60
    $elapsed = 0

    while ($true) {
        try {
            $client = New-Object System.Net.Sockets.TcpClient
            $client.Connect($hostname,$port)
            $client.Close()
            break
        }
        catch {
            Start-Sleep -Seconds 1
            $elapsed++

            if ($elapsed -ge $timeout) {
                throw "Timeout waiting for ${hostname}:${port}"
            }
        }
    }

    Write-Host "${hostname}:${port} is ready."
}

function Safe-StopProcess($processId) {
    try {
        if ($processId) {
            taskkill /PID $processId /T /F | Out-Null
        }
    } catch {}
}

function Start-Services {

    try {

        Write-Host ""
        Write-Host "Starting MongoDB..."

        docker start impactpoint-mongo 2>$null

        if ($LASTEXITCODE -ne 0) {
            docker run -d `
                --name impactpoint-mongo `
                -p 27017:27017 `
                mongo
        }

        Wait-ForPort "localhost" 27017


        Write-Host ""
        Write-Host "Starting Algorithm Service..."

        $algo = Start-Process powershell `
            -ArgumentList "-NoExit","-Command","cd services/algorithm_service; python -m uvicorn main:app --reload" `
            -PassThru

        Wait-ForPort "localhost" 8000


        Write-Host ""
        Write-Host "Starting Main Service..."

        $main = Start-Process powershell `
            -ArgumentList "-NoExit","-Command","cd services/main-service; npm run dev" `
            -PassThru

        Wait-ForPort "localhost" 3000


        Write-Host ""
        Write-Host "Starting Frontend..."

        $front = Start-Process powershell `
            -ArgumentList "-NoExit","-Command","cd frontend; ng serve" `
            -PassThru


        $processes = @{
            algorithm = $algo.Id
            main = $main.Id
            frontend = $front.Id
        }

        $processes | ConvertTo-Json | Set-Content $pidFile

        Write-Host ""
        Write-Host "🚀 All services started."

    }
    catch {
        Write-Host ""
        Write-Host "ERROR starting services:"
        Write-Host $_
    }
}

function Stop-Services {

    try {

        if (Test-Path $pidFile) {

            $pids = Get-Content $pidFile | ConvertFrom-Json

            Write-Host "Stopping services..."

            Safe-StopProcess $pids.algorithm
            Safe-StopProcess $pids.main
            Safe-StopProcess $pids.frontend

            Remove-Item $pidFile -ErrorAction SilentlyContinue
        }

        docker stop impactpoint-mongo 2>$null

        Write-Host "All services stopped."

    }
    catch {
        Write-Host "ERROR stopping services:"
        Write-Host $_
    }
}

function Status-Services {

    Write-Host ""
    Write-Host "Docker container:"

    docker ps | Select-String impactpoint-mongo

    if (Test-Path $pidFile) {

        $pids = Get-Content $pidFile | ConvertFrom-Json

        Write-Host ""
        Write-Host "Services running:"
        Write-Host "Algorithm PID:" $pids.algorithm
        Write-Host "Main PID:" $pids.main
        Write-Host "Frontend PID:" $pids.frontend

    } else {
        Write-Host ""
        Write-Host "No services running."
    }
}

switch ($action) {

    "start" { Start-Services }

    "stop" { Stop-Services }

    "restart" {
        Stop-Services
        Start-Sleep 2
        Start-Services
    }

    "status" { Status-Services }

    default {
        Write-Host ""
        Write-Host "Usage:"
        Write-Host "powershell ./dev.ps1 start"
        Write-Host "powershell ./dev.ps1 stop"
        Write-Host "powershell ./dev.ps1 restart"
        Write-Host "powershell ./dev.ps1 status"
    }
}