# Module 4: Memory & Conversations

By default, each `chat()` call is stateless — the agent doesn't remember previous turns. `Memory` adds conversation history automatically.

## Basic memory

```python
from lazybridge import LazyAgent, Memory

ai = LazyAgent("anthropic")
mem = Memory()

ai.chat("My name is Marco", memory=mem)
ai.chat("I live in London", memory=mem)
resp = ai.chat("What's my name and where do I live?", memory=mem)
print(resp.content)
# Your name is Marco and you live in London.
```

Without `memory=mem`, the agent would have no idea who you are.

## How it works

`Memory` accumulates turns internally:

```python
mem = Memory()
ai.chat("Hello", memory=mem)
ai.chat("How are you?", memory=mem)

# Inspect the history
for msg in mem.history:
    print(f"{msg['role']}: {msg['content'][:50]}")
# user: Hello
# assistant: Hello! How can I help you today?
# user: How are you?
# assistant: I'm doing well, thanks for asking!
```

## Shared memory across agents

The same Memory object can be passed to different agents — they share the conversation:

```python
translator = LazyAgent("anthropic", system="You are a translator. Translate to Italian.")
reviewer = LazyAgent("openai", system="You review translations for accuracy.")

mem = Memory()
translator.chat("Translate: 'The weather is beautiful today'", memory=mem)
reviewer.chat("Is the previous translation correct?", memory=mem)
print(reviewer.result)
```

## Persisting memory

Serialize memory for cross-session persistence:

```python
import json
from lazybridge import LazyAgent, Memory, LazyStore

# Session 1: build memory
store = LazyStore()
mem = Memory()
ai = LazyAgent("anthropic")
ai.chat("Remember: my favorite color is blue", memory=mem)
store.write("chat_history", mem.history)

# Session 2: restore memory
restored_history = store.read("chat_history")
mem2 = Memory.from_history(restored_history)
ai2 = LazyAgent("anthropic")
resp = ai2.chat("What's my favorite color?", memory=mem2)
print(resp.content)  # Blue!
```

## Thread safety

`Memory` is thread-safe — you can share it across threads for concurrent access:

```python
import threading

mem = Memory()
ai = LazyAgent("anthropic")

def worker(question):
    ai.chat(question, memory=mem)

threads = [threading.Thread(target=worker, args=(f"Question {i}",)) for i in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Total turns: {len(mem.history)}")  # 10 (5 questions + 5 answers)
```

---

## Exercise

1. Build a simple chatbot loop that remembers the conversation
2. Ask the agent personal questions across multiple turns and verify it remembers
3. Save the memory to a file and restore it in a new script

**Next:** [Module 5: Context Injection](05-context.md) — compose context from multiple sources.
