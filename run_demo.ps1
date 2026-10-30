# One-command demo runner.
#   .\run_demo.ps1                  # full pipeline + preview
#   .\run_demo.ps1 -NoRender        # pipeline only (about 2 seconds)
#   .\run_demo.ps1 -Wav D:\a.mp3    # your own track
param(
    [string]$Wav = "",
    [string]$Out = "out",
    [switch]$NoRender,
    [switch]$Planner,
    [double]$Difficulty = 0.6
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Pick an interpreter that actually has numpy + Pillow (>= 3.10).
$candidates = @(
    @{ exe = "py";  args = @("-3.14") },
    @{ exe = "py";  args = @("-3") },
    @{ exe = "python"; args = @() }
)
$chosen = $null
foreach ($c in $candidates) {
    if (-not (Get-Command $c.exe -ErrorAction SilentlyContinue)) { continue }
    $probe = & $c.exe @($c.args + @("-c", "import sys, numpy, PIL; print(sys.version_info[0]*100+sys.version_info[1])")) 2>$null
    if ($LASTEXITCODE -eq 0 -and [int]$probe -ge 310) { $chosen = $c; break }
}
if (-not $chosen) {
    throw "need a Python >= 3.10 with numpy and Pillow (tried py -3.14, py -3, python)"
}

$argsList = @("-m", "mmdance.cli", "demo", "--out", $Out, "--difficulty", $Difficulty)
if ($Wav) { $argsList += @("--wav", $Wav) }
if ($NoRender) { $argsList += "--no-render" }
if ($Planner) { $argsList += @("--planner", "llm") }

Write-Host ">> $($chosen.exe) $($chosen.args -join ' ') $($argsList -join ' ')" -ForegroundColor Cyan
& $chosen.exe @($chosen.args + $argsList)
exit $LASTEXITCODE
