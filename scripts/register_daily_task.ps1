<#
.SYNOPSIS
    Register the SarmayaSaaz data refresh with Windows Task Scheduler.

.DESCRIPTION
    Registers one task per market, each firing shortly after that market's
    session ends, so every asset class publishes its close the same day rather
    than waiting for a single overnight batch.

      16:15 PKT  PSX          regular board closes 15:30 PKT
      19:45 PKT  MUFAP        NAV publication is effectively complete by 19:00
      06:00 PKT  commodities  US settle ~21:00-22:00 UTC = ~02:00 PKT next day
                 + crypto     UTC day rolls at 00:00 UTC = 05:00 PKT next day

    Commodities and crypto cannot publish "same day" in any meaningful sense -
    their trading day ends after midnight Pakistan time - so the 06:00 run is
    already the first moment their previous session is final.

    Each task invokes the project venv's python.exe directly rather than
    `uv run`, because scheduled tasks get a much leaner environment than an
    interactive shell and uv may not be on the SYSTEM PATH.

    Every run ends with a snapshot rebuild, so the dashboard reflects whichever
    class just refreshed.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\register_daily_task.ps1
    powershell -ExecutionPolicy Bypass -File scripts\register_daily_task.ps1 -Remove
#>
param(
    [switch]$Remove
)

$ErrorActionPreference = "Stop"

$root   = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "scripts\daily_update.py"

if (-not $Remove) {
    if (-not (Test-Path $python)) { throw "venv python not found at $python - run 'uv sync --extra dev' first." }
    if (-not (Test-Path $script)) { throw "daily_update.py not found at $script" }
}

# Retired single-batch task from the first iteration; removed if still present.
$legacyName = "SarmayaSaaz Daily Update"

# Price tasks pass --skip-news: news collection walks ~123 assets with windowed
# queries and runs for hours, so it gets its own task rather than pushing every
# price refresh past the execution limit.
$tasks = @(
    @{ Name = "SarmayaSaaz Refresh (PSX)";    Args = "--only psx --skip-news";                At = "16:15"; Limit = 3;  Why = "PSX closes 15:30 PKT" },
    @{ Name = "SarmayaSaaz Refresh (MUFAP)";  Args = "--only mufap --skip-news";              At = "19:45"; Limit = 3;  Why = "NAV publication complete by 19:00 PKT" },
    @{ Name = "SarmayaSaaz Refresh (Global)"; Args = "--only commodities,crypto --skip-news"; At = "06:00"; Limit = 3;  Why = "previous UTC day is final by 05:00 PKT" },
    @{ Name = "SarmayaSaaz Refresh (News)";   Args = "";                                      At = "02:00"; Limit = 6;  Why = "slow and price-independent; runs at a quiet hour"; Script = "collect_news.py" }
)

foreach ($name in @($legacyName) + ($tasks | ForEach-Object { $_.Name })) {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
        Write-Host "Removed existing task '$name'"
    }
}

if ($Remove) {
    Write-Host "All SarmayaSaaz refresh tasks removed." -ForegroundColor Green
    return
}

# StartWhenAvailable: catch up a run missed while the machine was off.
# ExecutionTimeLimit: MUFAP walks ~200 funds with paced, retried requests.
function New-Settings([int]$hours) {
    New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit (New-TimeSpan -Hours $hours) `
        -MultipleInstances IgnoreNew
}

# S4U runs whether or not the user is logged on, without storing a password.
# It needs the "log on as a batch job" right, which a standard account often
# lacks - fall back to Interactive, which only runs while logged on.
function New-Principal([string]$logonType) {
    New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType $logonType -RunLevel Limited
}

$mode = $null
foreach ($t in $tasks) {
    $target = if ($t.Script) { Join-Path $root "scripts\$($t.Script)" } else { $script }
    $argline = if ($t.Args) { "`"$target`" $($t.Args)" } else { "`"$target`"" }
    $action  = New-ScheduledTaskAction -Execute $python -Argument $argline -WorkingDirectory $root
    $trigger = New-ScheduledTaskTrigger -Daily -At $t.At
    $settings = New-Settings $t.Limit
    $desc    = "$(Split-Path -Leaf $target) $($t.Args). Timing: $($t.Why). See docs/automation.md."

    try {
        Register-ScheduledTask -TaskName $t.Name -Action $action -Trigger $trigger `
            -Settings $settings -Principal (New-Principal "S4U") -Description $desc | Out-Null
        $mode = "S4U (runs whether or not you are logged on)"
    } catch {
        Register-ScheduledTask -TaskName $t.Name -Action $action -Trigger $trigger `
            -Settings $settings -Principal (New-Principal "Interactive") -Description $desc | Out-Null
        $mode = "Interactive (runs only while you are logged on)"
    }
    Write-Host ("Registered {0,-34} daily at {1}  limit {2}h" -f $t.Name, $t.At, $t.Limit) -ForegroundColor Green
}

Write-Host ""
Write-Host "mode   : $mode"
Write-Host "command: $python `"$script`" --only <classes>"
Write-Host "workdir: $root"
Write-Host "logs   : $root\logs\daily_update_<date>.log"
Write-Host ""
Write-Host "Run one now:  Start-ScheduledTask -TaskName 'SarmayaSaaz Refresh (PSX)'"
Write-Host "Remove all :  powershell -ExecutionPolicy Bypass -File scripts\register_daily_task.ps1 -Remove"
