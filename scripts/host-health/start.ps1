# Start host telemetry loops after a reboot. Each appends one row every 10 s to .scratch/host-health/*.csv,
# so a hard crash leaves the last readings on disk. LHM needs admin (UAC prompt) and LibreHardwareMonitor via winget.
param([switch]$NoLhm)
foreach ($s in 'gpu', 'cpu') {
  Start-Process -WindowStyle Hidden powershell.exe -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "$PSScriptRoot\${s}_telemetry.ps1"
}
if (-not $NoLhm) {
  Start-Process -Verb RunAs -WindowStyle Hidden powershell.exe -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "$PSScriptRoot\lhm_telemetry.ps1"
}
