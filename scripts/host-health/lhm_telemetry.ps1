# ponytail: load LibreHardwareMonitorLib directly (no GUI settings to keep in sync); needs admin for CPU/board sensors
param([switch]$List)
$lib = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\LibreHardwareMonitor*\LibreHardwareMonitorLib.dll" | Select -First 1
Add-Type -Path $lib.FullName
$pc = New-Object LibreHardwareMonitor.Hardware.Computer
$pc.IsCpuEnabled = $true; $pc.IsMotherboardEnabled = $true; $pc.IsMemoryEnabled = $true
$pc.Open()
$dir = "$PSScriptRoot\..\..\.scratch\host-health"
function All { foreach ($h in $pc.Hardware) { $h.Update(); $h; foreach ($s in $h.SubHardware) { $s.Update(); $s } } }
if ($List) {
  All | % { $h = $_; $h.Sensors | % { '{0} | {1} | {2} | {3}' -f $h.Name, $_.SensorType, $_.Name, $_.Value } } | Out-File -Encoding utf8 "$dir\lhm-sensors.txt"
  $pc.Close(); exit
}
$f = "$dir\lhm-telemetry.csv"
while ($true) {
  # ponytail: drop per-core rows and static limits so one line stays ~1 KB
  $v = All | % { $h = $_; $_.Sensors | ? { $_.SensorType -in 'Temperature', 'Power', 'Voltage', 'Fan' -and $_.Value -ne $null -and $_.Name -notmatch '[PE]-Core #|Limit|Resolution|Distance' } | % { '{0}/{1}={2:0.##}' -f $h.Name, $_.Name, $_.Value } }
  $v += ($pc.Hardware | ? HardwareType -eq 'Cpu' | % { $_.Sensors | ? { $_.Name -eq 'CPU Total' -or ($_.SensorType -eq 'Clock' -and $_.Name -eq 'P-Core #1') } | % { 'cpu/{0}={1:0.#}' -f $_.Name, $_.Value } })
  "$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss');$($v -join ';')" | Out-File -Append -Encoding utf8 $f
  Start-Sleep -Seconds 10
}
