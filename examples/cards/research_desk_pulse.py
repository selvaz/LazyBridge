"""Card 2: the same research desk, always on, in your Telegram.

LazyPulse keeps a lead agent (Claude Code, Opus) running. Its tools are the
research desk from Card 1 plus the two specialists. Message your bot any
research question; every weekday at 08:00 UTC it also sends a morning brief.

    export TELEGRAM_TOKEN=123456:ABC...   # from @BotFather
    export TELEGRAM_OWNER_ID=123456789    # your numeric Telegram id
    python research_desk_pulse.py
"""

import os

from lazypulse import Calendar, Cron, PulseAgent
from lazypulse.adapters.telegram import TelegramInbox, TelegramInboxConfig, TelegramPolicy
from lazytools.connectors.telegram import TelegramClient
from research_desk import desk, sonnet, terra

from lazybridge import ClaudeCodeEngine, Store

# Prompts
LEAD = (
    "You are an always-on research assistant. For a research question call "
    "research_desk. For a quick lookup call deep_research or fact_check. "
    "Reply in a few lines, with sources."
)
MORNING = "Brief me on the most important AI developer-tools news of the last 24 hours."

# Telegram: only the owner can talk to the bot
owner = int(os.environ["TELEGRAM_OWNER_ID"])
telegram = TelegramClient.from_token(os.environ["TELEGRAM_TOKEN"])

# Lead: Card 1's desk and the two specialists are all tools
pulse = PulseAgent(
    name="research_pulse",
    engine=ClaudeCodeEngine(model="opus", system=LEAD, web=False),
    tools=[desk, sonnet, terra],
    store=Store(db="research_pulse.db"),
    adapters=[TelegramInbox(telegram, TelegramInboxConfig(bot_id="research_pulse"))],
    policy=TelegramPolicy(owner_ids=[owner]),
    calendar=Calendar([Cron("morning", MORNING, "0 8 * * MON-FRI", notify=True)]),
    scheduled_responder=lambda text, *_: telegram.send_message(chat_id=owner, text=text[:4000]),
    max_concurrent_inbound=1,
)

if __name__ == "__main__":
    print("Always on. Message your bot; Ctrl-C to stop.")
    pulse.serve()
