$ErrorActionPreference='Stop'
$root='D:\IDD-PeD-Evaluation'; $archive="$root\archives\gp_set_0008.tar"; $total=[int64]2382336512; $log="$root\runs\autonomous_cpu_poc.log"
function Log([string]$t){"$(Get-Date -Format s) $t" | Tee-Object -FilePath $log -Append}
while (-not (Test-Path $archive) -or (Get-Item $archive).Length -lt $total) { Start-Sleep -Seconds 30 }
if ((Get-Item $archive).Length -ne $total) { throw 'Archive integrity size mismatch' }
Log 'Verified range-chunk set-0008 archive; extracting.'
& tar -xf $archive -C "$root\videos\gopro"
Log 'Extracted set-0008 videos; generating shared Farneback flows.'
& 'C:\Users\PC\Downloads\BTP\.venv\Scripts\python.exe' -u 'MTN\cpu_poc_prepare.py' 2>&1 | Tee-Object -FilePath $log -Append
Log 'Training and evaluating CPU POC.'
& 'C:\Users\PC\Downloads\BTP\.venv\Scripts\python.exe' -u 'MTN\cpu_poc_experiment.py' 2>&1 | Tee-Object -FilePath $log -Append
Log 'CPU POC complete.'
