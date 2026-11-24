$Host.UI.RawUI.WindowTitle = "Orchard Photos (200k)"
try {
    & (Join-Path $PSScriptRoot 'orchard.cmd') @args
}
finally {
    $env:ORCHARD_POWERSHELL_CONSOLE = $null
}
