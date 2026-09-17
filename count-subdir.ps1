param(
    [Parameter(Mandatory = $true)]
    [string]$TargetDirectory
)

$total = 0
$fileCount = 0

Write-Host ""
Write-Host "Directory:"
Write-Host $TargetDirectory
Write-Host ""

Get-ChildItem -LiteralPath $TargetDirectory -Recurse -File -Filter *.pdf | ForEach-Object {
    Write-Host "Processing: $($_.FullName)"

    $info = & pdfinfo $_.FullName 2>$null
    $match = $info | Select-String '^Pages:\s+(\d+)'

    if ($match) {
        $pages = [int]$match.Matches.Groups[1].Value
        $total += $pages
        $fileCount++
        Write-Host "  $pages pages   Total: $total"
    }
    else {
        Write-Host "  FAILED" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "===================================="
Write-Host "PDF files:   $fileCount"
Write-Host "Total pages: $total"
