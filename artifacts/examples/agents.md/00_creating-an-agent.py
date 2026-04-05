# Source   : lazy_wiki/human/agents.md
# Heading  : Creating an agent
# ID       : lazy_wiki/human/agents.md::creating-an-agent::00
# Kind     : llm_chat
# Testable : smoke_exec

from lazybridge import LazyAgent

# Minimal — just pick a provider
ai = LazyAgent("anthropic")

# With options
ai = LazyAgent(
    "anthropic",
    name="my_assistant",           # used in logs and when exposing as a tool
    model="claude-sonnet-4-6",     # override provider default
    system="You are a terse assistant. Answer in one sentence.",
    max_retries=3,                 # retry on rate limits / server errors
)
resp = ai.chat("What is 2 + 2?")
print(resp.content)       # "4"
print(resp.usage.input_tokens)
print(resp.stop_reason)   # "end_turn"


from lazybridge import LazyAgent, Memory

ai  = LazyAgent("gemini")
mem = Memory()

ai.chat("My name is Marco", memory=mem)
resp = ai.chat("What's my name?", memory=mem)
print(resp.content)   # "Marco"


print(len(mem))        # 4 — 2 user + 2 assistant messages
print(mem.history)     # list of {"role": ..., "content": ...} dicts



from lazybridge import LazyAgent, Memory

mem = Memory()
agent_a = LazyAgent("anthropic")
agent_b = LazyAgent("anthropic")
agent_a.chat("Remember: the project deadline is Friday", memory=mem)
agent_b.chat("What's the deadline?", memory=mem)   # answers "Friday"





import json
from lazybridge import LazyAgent, Memory, LazySession

sess = LazySession(db="chat.db")
ai  = LazyAgent("anthropic", session=sess)

# Restore previous session
raw = sess.store.read("history")
mem = Memory.from_history(json.loads(raw)) if raw else Memory()

ai.chat("What is my name?", memory=mem)

# Save at the end
sess.store.write("history", json.dumps(mem.history))
