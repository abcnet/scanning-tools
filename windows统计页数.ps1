Set-Location (Split-Path $PSScriptRoot -Parent)

$total = 0

Get-ChildItem -Recurse -File -Filter *.pdf | ForEach-Object {
    Write-Host "Processing: $($_.FullName)"

    $info = & pdfinfo $_.FullName 2>$null
    $match = $info | Select-String '^Pages:\s+(\d+)'

    if ($match) {
        $pages = [int]$match.Matches.Groups[1].Value
        $total += $pages
        Write-Host "  $pages pages   Total: $total"
    }
    else {
        Write-Host "  FAILED" -ForegroundColor Red
    }
}

Write-Host "`nTotal pages: $total"