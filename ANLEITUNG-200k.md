# Orchard Photos (200k-Version) — Anleitung & PowerShell-Befehle

Diese Version von Orchard Photos basiert auf `chaseleantj/orchard-photos` und wurde im Python-Backend für **200.000 Bilder** optimiert.

---

## 1. Voraussetzungen (Windows, Running from Source)

Gemäß `docs/running-from-source.md` benötigt Orchard Python 3.12, 3.13 oder 3.14 sowie das Microsoft Visual C++ Redistributable (x64):

```powershell
winget install Python.Python.3.13
winget install Microsoft.VCRedist.2015+.x64
```

Nach dem Entpacken von `orchard-photos-200k.zip` installiert der erste Start von `.\orchard.cmd` automatisch alle benötigten Pakete in `.venv`.

---

## 2. Suchmodell (SigLIP 2) per PowerShell herunterladen

Falls der automatische Download über `huggingface_hub` in der App unter Windows abbricht (z. B. durch Echtzeitschutz / Symlink-Rechte), kannst du das mitgelieferte Skript `.\download-model.ps1` ausführen oder den folgenden PowerShell-Befehl direkt nutzen:

```powershell
# 1. Zielordner für das SigLIP-2-Modell anlegen (exakte Revision aus server/semantic.py)
$RepoDir = "$env:LOCALAPPDATA\orchard\models\models--onnx-community--siglip2-base-patch16-256-ONNX"
$ModelDir = "$RepoDir\snapshots\d1114256522a37ffa257a0a58017348ab0058db2"
New-Item -ItemType Directory -Force -Path "$ModelDir\onnx" | Out-Null

# 2. Unvollständige Downloads aufräumen
Remove-Item -Path "$RepoDir\blobs\*.incomplete" -Force -ErrorAction SilentlyContinue

# 3. Alle 3 Modelldateien direkt mit curl.exe herunterladen (-C - setzt abgebrochene Downloads fort)
$BaseUrl = "https://huggingface.co/onnx-community/siglip2-base-patch16-256-ONNX/resolve/d1114256522a37ffa257a0a58017348ab0058db2"

curl.exe -fL --retry 5 -C - "$BaseUrl/tokenizer.json" -o "$ModelDir\tokenizer.json"
curl.exe -fL --retry 5 -C - "$BaseUrl/onnx/text_model_int8.onnx" -o "$ModelDir\onnx\text_model_int8.onnx"
curl.exe -fL --retry 5 -C - "$BaseUrl/onnx/vision_model.onnx" -o "$ModelDir\onnx\vision_model.onnx"
```

---

## 3. Modell & Engine (DirectML / CPU) prüfen

```powershell
.\.venv\Scripts\python.exe -c "from server.api import default_data_dir; from server.semantic import Encoder; e = Encoder(default_data_dir() / 'models'); print('Vollständig heruntergeladen:', e.downloaded()); e.load(); print('Modell erfolgreich geladen auf:', e.engine)"
```

---

## 4. Orchard starten

```powershell
# Im eigenen App-Fenster starten:
.\orchard.cmd

# Direkt mit einem Bilder-Ordner starten:
.\orchard.cmd "C:\Pfad\Zu\Deinen\Bildern"

# Im Browser (http://127.0.0.1:4400) öffnen:
.\orchard.cmd "C:\Pfad\Zu\Deinen\Bildern" --browser

# Für sehr große Ordner / Netzlaufwerke (ohne automatischen 60-Sekunden-Rescan):
.\orchard.cmd "C:\Pfad\Zu\Deinen\Bildern" --no-watch
```

---

## Was für 200.000 Bilder optimiert wurde

- **`server/library.py`**: `MAP_LIMIT` von `25_000` auf `200_000` erhöht; `facets()` revisionsbasiert gecacht, sodass bei 200k Bildern keine unnötigen Voll-Abfragen bei jedem Status-Poll ausgeführt werden.
- **`server/semantic.py`**: SQLite-Index `semantic_embeddings_model` auf `semantic_embeddings(model)` ergänzt; Laden von 200k Vektoren in `embeddings()` über einen zusammenhängenden Buffer optimiert; Thumbnail-Öffnen bei der Indizierung parallelisiert (`ThreadPoolExecutor`).
- **`server/groups.py`**: Ab `> 10.000` Bildern (`EXACT_LIMIT`) nutzt `gather` ein Spherical-k-Means-Cluster-Pruning analog zu `server/layout.py`, statt eine 200k × 200k Ähnlichkeitsmatrix komplett zu multiplizieren.
- **`server/atlas.py`**: `Atlas.ensure()` rendert und schreibt Map-Icon-Sheets gebatcht pro Sheet (`PER_SHEET = 1024`), statt 200.000 Icons gleichzeitig im RAM zu halten.
