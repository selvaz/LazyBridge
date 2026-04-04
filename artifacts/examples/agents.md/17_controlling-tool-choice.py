# Source   : lazy_wiki/human/agents.md
# Heading  : Controlling tool choice
# ID       : lazy_wiki/human/agents.md::controlling-tool-choice::00
# Kind     : llm_loop
# Testable : smoke_exec

# Force the model to call at least one tool
result = ai.loop("Summarise today's news", tools=[search], tool_choice="required")

# Prevent any tool calls
resp = ai.chat("Just answer from memory", tools=[search], tool_choice="none")

# Force a specific tool
result = ai.loop("Find news", tools=[search, calc], tool_choice="search_web")
