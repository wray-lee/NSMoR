# ponytail: native counters only (no driver install); CPU temp/package power need LibreHardwareMonitor/HWiNFO, user's call
$f = "$PSScriptRoot\..\..\.scratch\host-health\cpu-telemetry.csv"
$c = '\Processor Information(_Total)\% Processor Utility', '\Processor Information(_Total)\% Processor Performance', '\Memory\Available MBytes'
while ($true) {
  $v = (Get-Counter $c -ErrorAction SilentlyContinue).CounterSamples.CookedValue | ForEach-Object { [math]::Round($_, 1) }
  "$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'),$($v -join ',')" | Out-File -Append -Encoding ascii $f
  Start-Sleep -Seconds 10
}
