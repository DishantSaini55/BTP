$ErrorActionPreference = 'Stop'
$root = 'D:\IDD-PeD-Evaluation'
$target = "$root\archives\gp_set_0008.tar"
$chunk = "$root\archives\gp_set_0008.chunk"
$url = 'https://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_0008.tar'
$total = [int64]2382336512
$chunkSize = [int64](64MB)
$log = "$root\runs\set0008_chunk_download.log"
function Log([string]$t) { "$(Get-Date -Format s) $t" | Tee-Object -FilePath $log -Append }
if (Test-Path $target) { Remove-Item -LiteralPath $target -Force }
if (Test-Path $chunk) { Remove-Item -LiteralPath $chunk -Force }
New-Item -ItemType File -Path $target -Force | Out-Null
$offset = [int64]0
while ($offset -lt $total) {
  $end = [Math]::Min($offset + $chunkSize - 1, $total - 1)
  $expected = $end - $offset + 1
  $ok = $false
  for ($attempt = 1; $attempt -le 10 -and -not $ok; $attempt++) {
    Remove-Item -LiteralPath $chunk -Force -ErrorAction SilentlyContinue
    Log "range $offset-$end attempt $attempt"
    & curl.exe -L --fail --retry 2 --retry-all-errors --retry-delay 3 --range "$offset-$end" -o $chunk $url
    if ((Test-Path $chunk) -and (Get-Item $chunk).Length -eq $expected) { $ok = $true } else { Start-Sleep -Seconds 3 }
  }
  if (-not $ok) { throw "Could not obtain verified range $offset-$end" }
  $source=[IO.File]::OpenRead($chunk); $destination=[IO.File]::Open($target,[IO.FileMode]::Append,[IO.FileAccess]::Write)
  try { $source.CopyTo($destination) } finally { $source.Dispose(); $destination.Dispose() }
  $offset += $expected
  if ((Get-Item $target).Length -ne $offset) { throw 'Target byte count mismatch' }
}
Remove-Item -LiteralPath $chunk -Force
Log "completed $((Get-Item $target).Length) bytes"
