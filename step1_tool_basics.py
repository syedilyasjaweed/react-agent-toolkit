from dotenv import load_dotenv
load_dotenv()

import anthropic

client = anthropic.Anthropic()

def get_weather(city):
    fake_weather_data = {
        "Chicago": "68°F, partly cloudy",
        "New York": "72°F, sunny",
        "Los Angeles": "80°F, clear skies"
    }
    return fake_weather_data.get(city, f"No weather data available for {city}")

def get_time(city):
    fake_time_data = {
        "Chicago": "2:15 PM CDT",
        "New York": "3:15 PM EDT",
        "Los Angeles": "12:15 PM PDT"
    }
    return fake_time_data.get(city, f"No time data available for {city}")

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a given city",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name, e.g. 'Chicago'"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_time",
        "description": "Get the current local time for a given city",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name, e.g. 'Chicago'"}
            },
            "required": ["city"]
        }
    }
]

def run_tool(name, tool_input):
    if name == "get_weather":
        return get_weather(tool_input["city"])
    elif name == "get_time":
        return get_time(tool_input["city"])
    else:
        return f"Unknown tool: {name}"

# --- The ReAct loop ---
messages = [
    {"role": "user", "content": "What's the weather and time in Chicago right now?"}
]

while True:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        tools=tools,
        messages=messages
    )

    # Add Claude's response to the conversation
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason != "tool_use":
        # Claude is done — print final answer and exit loop
        final_text = next(block.text for block in response.content if block.type == "text")
        print("\n--- Final answer ---")
        print(final_text)
        break

    # Claude wants to use one or more tools — handle each one
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            print(f"\n--- Claude is calling: {block.name} with {block.input} ---")
            result = run_tool(block.name, block.input)
            print(f"--- Result: {result} ---")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result
            })

    # Feed all tool results back as the next user turn
    messages.append({"role": "user", "content": tool_results})
