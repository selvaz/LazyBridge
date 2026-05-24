# LazyPulse — always-on orchestration

**Give an LLM agent a heartbeat.** LazyPulse turns a one-shot agent into an
always-on one: it watches an inbox / webhook / queue, decides *who is allowed to
ask it for what*, runs the work in the background, and pauses for your approval
before anything risky (like sending an email) actually happens.

A `PulseAgent` **is** a `lazybridge.Agent` with three additions — a **tick
loop**, a **trust policy**, and **inbound adapters**:

```
   inbound message            PulsePolicy                 your Agent
  ┌──────────────┐   drain   ┌────────────┐   allow?   ┌────────────┐
  │ Gmail        │ ────────> │ who sent    │ ────────> │ engine +   │
  │ Webhook      │           │ this? what  │  review?  │ tools +    │
  │ your adapter │           │ may they    │  reject?  │ verify     │
  └──────────────┘           │ ask for?    │           └────────────┘
        every tick_seconds    └────────────┘            lifecycle in Store
```

```bash
pip install lazypulse                    # core tick loop + policy
pip install 'lazypulse[gmail]'           # Gmail inbox (pulls lazytoolkit[gmail])
pip install 'lazypulse[telegram]'        # Telegram inbox (pulls lazytoolkit[telegram])
pip install 'lazypulse[webhook]'         # HTTP intake adapter
```

## How it relates to the other two packages

- It builds on **lazybridge**: `PulseAgent` subclasses `Agent`, so the full
  Agent surface (engine, tools, guard, verify, memory, store, session) works
  unchanged.
- It uses **lazytools** for capabilities: the Gmail/Telegram **inbound
  adapters** (inbox + trust policy) live in LazyPulse, while the matching
  **clients and guarded send tools** live in `lazytools.connectors.*`. Installing
  `lazypulse[gmail]` pulls `lazytoolkit[gmail]` for you.

The division of labour: a **Tool** the worker invokes mid-run lives in
`lazytools`; an **inbound adapter / policy** that produces messages and decides
trust lives in `lazypulse`.

Repo & full docs: <https://github.com/selvaz/LazyPulse>
