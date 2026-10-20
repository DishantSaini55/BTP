$ErrorActionPreference = 'Stop'
$root = 'D:\IDD-PeD-Evaluation'
$archive = "$root\archives\gp_set_0008.tar"
$url = 'https://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0008.tar'
$total = 2382336512
$log = "$root\runs\autonomous_cpu_poc.log"
function Log([string]$text) { "$(Get-Date -Format s) $text" | Tee-Object -FilePath $log -Append }
Log 'Autonomous CPU POC pipeline started.'
while ($true) {
    $size = if (Test-Path -LiteralPath $archive) { (Get-Item -LiteralPath $archive).Length } else { 0 }
    if ($size -eq $total) { break }
    if ($size -gt $total) { throw "Archive exceeds official size: $size > $total" }
    Log "Downloading/resuming set 0008 at $size bytes."
    & curl.exe -L --fail --retry 5 --retry-all-errors --retry-delay 5 --speed-time 45 --speed-limit 1 --continue-at - -o $archive $url
    Start-Sleep -Seconds 5
}
Log 'Verified set-0008 archive size.'
& tar -tf $archive | Select-Object -First 3 | Out-File "$root\runs\set0008_archive_listing.txt"
New-Item -ItemType Directory -Force -Path "$root\videos\gopro" | Out-Null
& tar -xf $archive -C "$root\videos\gopro"
Log 'Extracted set-0008 videos.'
& 'C:\Users\PC\Downloads\BTP\.venv\Scripts\python.exe' -u 'MTN\cpu_poc_prepare.py' 2>&1 | Tee-Object -FilePath $log -Append
Log 'Generated fixed manifest and shared Farneback flows.'
& 'C:\Users\PC\Downloads\BTP\.venv\Scripts\python.exe' -u 'MTN\cpu_poc_experiment.py' 2>&1 | Tee-Object -FilePath $log -Append
Log 'CPU POC complete.'
