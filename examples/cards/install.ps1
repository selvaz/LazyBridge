# LazyBridge cards installer (Windows PowerShell).
#   irm https://raw.githubusercontent.com/selvaz/LazyBridge/main/examples/cards/install.ps1 | iex
# Installs into $HOME\lazybridge-cards: a Python venv with LazyBridge + LazyPulse,
# the Claude Code and Codex CLIs if missing, and the two card scripts.
# It never asks for or stores credentials: you sign in yourself at the end.
$ErrorActionPreference = "Stop"

$Dir = if ($env:LAZYBRIDGE_CARDS_DIR) { $env:LAZYBRIDGE_CARDS_DIR } else { Join-Path $HOME "lazybridge-cards" }
$Raw = "https://raw.githubusercontent.com/selvaz/LazyBridge/main/examples/cards"
$LazyBridge = "lazybridge[claude-code]==1.8.7"
$LazyPulse = "lazypulse[telegram,cron] @ git+https://github.com/selvaz/LazyPulse.git@v0.5.7"

function Say($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Has($c) { [bool](Get-Command $c -ErrorAction SilentlyContinue) }

Say "Checking Python 3.11+ and git"
$Py = $null
foreach ($c in @("py -3.13", "py -3.12", "py -3.11", "python")) {
    $exe, $arg = $c.Split(" ")
    if (Has $exe) {
        & $exe $arg -c "import sys; sys.exit(sys.version_info < (3, 11))" 2>$null
        if ($LASTEXITCODE -eq 0) { $Py = @($exe) + @($arg | Where-Object { $_ }); break }
    }
}
if (-not $Py) { throw "Python 3.11+ not found: winget install Python.Python.3.12, then rerun." }
if (-not (Has git)) { throw "git not found: winget install Git.Git, then rerun." }

Say "Creating $Dir and its Python environment"
New-Item -ItemType Directory -Force $Dir | Out-Null
$VPy = Join-Path $Dir ".venv\Scripts\python.exe"
if (-not (Test-Path $VPy)) { & $Py[0] $Py[1..9] -m venv (Join-Path $Dir ".venv") }
& $VPy -m pip install --quiet --upgrade pip
& $VPy -m pip install --quiet $LazyBridge $LazyPulse
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

Say "Claude Code CLI"
if (Has claude) { "already installed" } else { irm https://claude.ai/install.ps1 | iex }

Say "Codex CLI"
if (Has codex) { "already installed" }
elseif (Has npm) { npm install -g @openai/codex }
else { "npm not found: winget install OpenJS.NodeJS.LTS, then run: npm install -g @openai/codex" }

Say "Downloading the cards"
foreach ($f in @("research_desk.py", "research_desk_pulse.py")) {
    Invoke-WebRequest "$Raw/$f" -OutFile (Join-Path $Dir $f) -UseBasicParsing
}

@"

Done. Next:
  1. Sign in once:   claude        (then /exit)
                     codex login
  2. Card 1:         cd "$Dir"; .venv\Scripts\python research_desk.py "your question"
  3. Card 2:         `$env:TELEGRAM_TOKEN="..."; `$env:TELEGRAM_OWNER_ID="..."
                     cd "$Dir"; .venv\Scripts\python research_desk_pulse.py
  Quick check of every engine:  .venv\Scripts\python -m lazybridge.demo
"@
