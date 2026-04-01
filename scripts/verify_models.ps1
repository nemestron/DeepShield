param(
    [string]$ModelsPath = "models"
)

$requiredModels = @("image_model.h5", "audio_model.h5")
$missingModels = @()

Write-Host "`n--- DeepShield Model Verification Protocol ---" -ForegroundColor Cyan

if (-not (Test-Path -Path $ModelsPath)) {
    Write-Error "CRITICAL: Models directory not found at path: $ModelsPath"
    exit 1
}

foreach ($model in $requiredModels) {
    $fullPath = Join-Path -Path $ModelsPath -ChildPath $model
    if (Test-Path -Path $fullPath) {
        $fileInfo = Get-Item $fullPath
        $sizeMB = [math]::Round($fileInfo.Length / 1MB, 2)
        Write-Host "[OK] Found $model ($sizeMB MB)" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Missing $model" -ForegroundColor Red
        $missingModels += $model
    }
}

if ($missingModels.Count -gt 0) {
    Write-Host "`n[STATUS: FAILED] The following model artifacts are missing:" -ForegroundColor Red
    foreach ($missing in $missingModels) {
        Write-Host " -> $missing" -ForegroundColor Red
    }
    Write-Host "`nACTION REQUIRED: Download the missing .h5 files from Google Colab and place them inside the '$ModelsPath' directory." -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "`n[STATUS: PASSED] All model artifacts verified. Ready for local integration." -ForegroundColor Green
    exit 0
}