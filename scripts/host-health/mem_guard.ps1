# ponytail: kd/cdb on the 6.8 GB MEMORY.DMP drained ~21 GB of Windows RAM in ~40 s three times on 11-15 and hung the host.
# Kill debugger processes when Windows available memory gets low; nothing else is touched.
$log = "$PSScriptRoot\..\..\.scratch\host-health\mem-guard.log"
while ($true) {
  $mb = [int](Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue
  if ($mb -lt 6000) {
    Get-Process kd, cdb, windbg, DbgX.Shell, EngHost -ErrorAction SilentlyContinue | ForEach-Object {
      "$(Get-Date -Format s) avail=${mb}MB kill $($_.ProcessName) $($_.Id)" | Out-File -Append -Encoding ascii $log
      Stop-Process -Id $_.Id -Force
    }
  }
  Start-Sleep -Seconds 1
}
