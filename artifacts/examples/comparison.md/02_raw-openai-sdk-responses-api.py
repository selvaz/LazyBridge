# Source   : lazy_wiki/human/comparison.md
# Heading  : Raw OpenAI SDK (Responses API)
# ID       : lazy_wiki/human/comparison.md::raw-openai-sdk-responses-api::00
# Kind     : local
# Testable : local_exec

import json
import openai

client = openai.OpenAI()

# 1. Define schema manually (Responses API flattened format)
tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"}
            },
            "required": ["city"],
        },
    }
]

def get_weather(city: str) -> str:
    return f"Weather in {city}: 22°C, sunny"

input_messages = [{"role": "user", "content": "What's the weather in Rome and Paris?"}]

# 2. Manual tool-call loop
while True:
    response = client.responses.create(
        model="gpt-4o",
        input=input_messages,
        tools=tools,
    )
    # Check for function calls in output items
    function_calls = [item for item in response.output if item.type == "function_call"]

    if function_calls:
        # Append assistant output items to conversation
        input_messages.extend([item.model_dump() for item in response.output])
        for fc in function_calls:
            args = json.loads(fc.arguments)
            result = get_weather(**args)
            input_messages.append({
                "type": "function_call_output",
                "call_id": fc.call_id,
                "output": result,
            })
    else:
        # Extract text from message output items
        text_items = [item for item in response.output if item.type == "message"]
        print(text_items[0].content[0].text)
        break
