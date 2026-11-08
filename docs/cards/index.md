# Cards

Each card is one short, runnable file that shows how LazyBridge composes
agents. Every agent is a tool of another agent, and engines are peers:
Claude Code, Codex and provider APIs plug in the same way.

| Card | What it shows |
|---|---|
| [Research desk](research-desk.md) | A lead agent whose tools are a deterministic pipeline of sub-agents plus the specialists themselves. Plain Python. |
| [Research desk, always on](research-desk-pulse.md) | The same desk running 24/7 in your Telegram through LazyPulse, with a morning brief. |

## Install everything with one command

=== "Windows (PowerShell)"

    ```powershell
    irm https://raw.githubusercontent.com/selvaz/LazyBridge/main/examples/cards/install.ps1 | iex
    ```

=== "macOS / Linux"

    ```console
    curl -fsSL https://raw.githubusercontent.com/selvaz/LazyBridge/main/examples/cards/install.sh | bash
    ```

The installer creates `~/lazybridge-cards` with its own Python environment,
installs LazyBridge, LazyPulse and the Claude Code and Codex CLIs if they are
missing, and downloads the cards. It never asks for credentials: you do three
things yourself, explained below: sign in to Claude Code and Codex, set a
DeepSeek API key and, for Card 2, create a Telegram bot.

## Sign in to Claude Code and Codex (required)

The cards run on the Claude Code and Codex CLIs installed on your machine.
They only work after you sign in to each one **once**, in a terminal:

1. **Claude Code:** run `claude`, follow the browser sign-in, then type `/exit`.
2. **Codex:** run `codex login` and follow the browser sign-in.
   `codex login status` tells you whether it worked.

With a Claude or ChatGPT subscription login, the engines use your plan's
quota. LazyBridge reuses this sign-in; it never sees or stores your password.

## Set the DeepSeek API key

The research desk's summary is written by DeepSeek over its API, billed per
token. Create a key at [platform.deepseek.com](https://platform.deepseek.com)
and set it in the terminal where you run the cards:

=== "Windows (PowerShell)"

    ```powershell
    $env:DEEPSEEK_API_KEY = "sk-..."
    ```

=== "macOS / Linux"

    ```console
    export DEEPSEEK_API_KEY="sk-..."
    ```

## Set up the Telegram bot (Card 2 only)

1. In Telegram, open [@BotFather](https://t.me/BotFather), send `/newbot`,
   choose a name and a username. BotFather replies with a **token** such as
   `123456789:AAH...`. Keep it secret: anyone with it controls the bot.
2. Send any message to your new bot (this lets it see you).
3. Open `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser, with
   your token in place of `<TOKEN>`. Find `"from":{"id":123456789,...}`:
   that number is your **owner id**. Only this id can talk to the desk.
4. Set both values in the terminal where you start Card 2:

=== "Windows (PowerShell)"

    ```powershell
    $env:TELEGRAM_TOKEN = "123456789:AAH..."
    $env:TELEGRAM_OWNER_ID = "123456789"
    ```

=== "macOS / Linux"

    ```console
    export TELEGRAM_TOKEN="123456789:AAH..."
    export TELEGRAM_OWNER_ID="123456789"
    ```

These variables last for that terminal session only. Never commit the token
to a repository.

Not sure which engines work on your machine? Run the two-minute check:

```console
python -m lazybridge.demo
```
