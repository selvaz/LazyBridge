# For Beginners

**You know Python. You've heard about LLMs. You want to build something — but you don't know where to start.**

This section is for you.

No prior experience with LLM APIs, agent frameworks, or "AI infrastructure" is required.
By the end you'll have built a working multi-agent pipeline from scratch, and you'll
understand *why* every piece exists.

---

## What you'll learn

```
Step 1 → What is an LLM and how does an API call actually work?
Step 2 → What does raw SDK code look like? (OpenAI / Anthropic / Gemini)
Step 3 → Your first LazyBridge agent — every line explained
Step 4 → Giving your agent tools (so it can actually do things)
Step 5 → Multiple agents working together
Step 6 → LazyBridge vs LangGraph vs CrewAI — which one for what
Step 7 → Where to go next
```

Each step builds on the previous one. Skip ahead if you already know a topic,
but the examples in later steps assume you've seen the earlier ones.

---

## Prerequisites

- **Python 3.10+** installed and working
- You can write a function, use a `for` loop, and read a stack trace
- That's it

You do **not** need to know: machine learning, transformers, embeddings, vector databases,
async programming, or any other AI/ML concept. We'll introduce everything as we need it.

---

## What LazyBridge is (one paragraph)

LazyBridge is a Python library that lets you build LLM-powered agents with the minimum
amount of code. An **agent** is a program that uses an LLM to decide what to do — it can
call functions you write, talk to other agents, remember context across turns, and return
structured results. LazyBridge handles all the plumbing (API calls, retries, tool dispatch,
result parsing) so you write business logic, not boilerplate.

---

## What you'll build by the end

A pipeline that takes a research question, searches the web, summarises the findings,
and returns a one-paragraph report — using three specialised agents that hand off work
to each other automatically.

!!! note "API keys"
    The examples in this tutorial use real LLM providers (Anthropic, OpenAI, Google).
    You'll need at least one API key. If you don't have one yet, sign up for the
    [Anthropic Console](https://console.anthropic.com/) — they offer free credits for
    new accounts, and LazyBridge defaults to Anthropic.

    Every example also has a **mock variant** that runs without any API key, so you can
    follow along even before you have credentials.

---

## Ready? Start here →

[**Step 1: What is an LLM?**](01-what-is-an-llm.md){ .md-button .md-button--primary }
