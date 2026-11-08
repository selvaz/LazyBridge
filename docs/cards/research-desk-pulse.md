# Card 2: Research desk, always on

**The same desk, alive 24/7 in your Telegram. Card 1's agent is one of its tools.**

[LazyPulse](https://github.com/selvaz/LazyPulse) keeps a lead agent running.
Message your bot any question and get a cited answer back in the chat. Every
weekday at 08:00 UTC it also sends you a morning brief on its own.

<div class="lb-card-figure" markdown="0">
--8<-- "docs/cards/research-desk-pulse.svg"
</div>

## How it is built

1. **Import Card 1.** `from research_desk import desk, sonnet, terra`: the
   whole desk is one object, and it becomes a tool.
2. **A `PulseAgent` is an `Agent` that never stops.** Same `engine=` and
   `tools=`, plus where work comes from (`adapters=`, `calendar=`) and who may
   ask (`policy=`).
3. **Telegram in, Telegram out.** `TelegramInbox` polls your bot and replies
   in the same chat; `TelegramPolicy` lets only you talk to it.
4. **A calendar.** `Cron("morning", ...)` fires a task on schedule;
   `notify=True` sends the result to you.

## The code

```python
--8<-- "examples/cards/research_desk_pulse.py"
```

## Run it

1. Create the bot and find your owner id: see [Set up the Telegram bot](index.md#set-up-the-telegram-bot-card-2-only).
2. Start it:

```console
export TELEGRAM_TOKEN=123456:ABC...
export TELEGRAM_OWNER_ID=123456789
python research_desk_pulse.py
```

It runs in the foreground; stop it with Ctrl-C.
