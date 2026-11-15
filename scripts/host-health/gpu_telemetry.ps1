# ponytail: append one nvidia-smi sample every 10 s so a hard crash leaves the last readings on disk
$f = "$PSScriptRoot\..\..\.scratch\host-health\gpu-telemetry.csv"
while ($true) {
  nvidia-smi.exe --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used,pstate,clocks_event_reasons.active --format=csv,noheader | Out-File -Append -Encoding ascii $f
  Start-Sleep -Seconds 10
}
