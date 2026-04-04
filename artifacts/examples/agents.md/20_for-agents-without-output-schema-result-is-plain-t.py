# Source   : lazy_wiki/human/agents.md
# Heading  : For agents without output_schema, result is plain text:
# ID       : lazy_wiki/human/agents.md::for-agents-without-output-schema-result-is-plain-text::00
# Kind     : local
# Testable : local_exec

resp = ai._last_response
print(resp.usage.input_tokens, resp.usage.output_tokens)
print(resp.stop_reason)
