$ErrorActionPreference = "Stop"

# Zielordner für das SigLIP-2-Modell (exakte Revision aus server/semantic.py)
$RepoDir = "$env:LOCALAPPDATA\orchard\models\models--onnx-community--siglip2-base-patch16-256-ONNX"
$ModelDir = "$RepoDir\snapshots\d1114256522a37ffa257a0a58017348ab0058db2"
New-Item -ItemType Directory -Force -Path "$ModelDir\onnx" | Out-Null

# Eventuell hängengebliebene unvollständige Downloads aufräumen
Remove-Item -Path "$RepoDir\blobs\*.incomplete" -Force -ErrorAction SilentlyContinue

# Alle 3 Modelldateien direkt mit curl.exe herunterladen (-C - setzt abgebrochene Downloads fort)
$BaseUrl = "https://huggingface.co/onnx-community/siglip2-base-patch16-256-ONNX/resolve/d1114256522a37ffa257a0a58017348ab0058db2"

Write-Host "Lade tokenizer.json (~34 MB)..." -ForegroundColor Cyan
curl.exe -fL --retry 5 -C - "$BaseUrl/tokenizer.json" -o "$ModelDir\tokenizer.json"
if ($LASTEXITCODE -ne 0) { throw "Download von tokenizer.json fehlgeschlagen." }

Write-Host "Lade text_model_int8.onnx (~283 MB)..." -ForegroundColor Cyan
curl.exe -fL --retry 5 -C - "$BaseUrl/onnx/text_model_int8.onnx" -o "$ModelDir\onnx\text_model_int8.onnx"
if ($LASTEXITCODE -ne 0) { throw "Download von text_model_int8.onnx fehlgeschlagen." }

Write-Host "Lade vision_model.onnx (~372 MB)..." -ForegroundColor Cyan
curl.exe -fL --retry 5 -C - "$BaseUrl/onnx/vision_model.onnx" -o "$ModelDir\onnx\vision_model.onnx"
if ($LASTEXITCODE -ne 0) { throw "Download von vision_model.onnx fehlgeschlagen." }

Write-Host "Alle 3 Modelldateien wurden heruntergeladen!" -ForegroundColor Green
