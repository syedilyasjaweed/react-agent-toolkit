from dotenv import load_dotenv
load_dotenv()  # reads .env so ANTHROPIC_API_KEY is available without hardcoding it

import anthropic

# This client is your connection to the Claude API. Every round of the loop
# below re-uses this same client to send a new request.
client = anthropic.Anthropic()


# --- Fake tools ---
# These are plain Python functions with no real API calls behind them.
# The point of Step 1 is to learn the tool-calling MECHANICS risk-free,
# before wiring in anything real (that comes in Step 2 and Step 3).

def get_weather(city):
    fake_weather_data = {
        "Chicago": "68°F, partly cloudy",
        "New York": "72°F, sunny",
        "Los Angeles": "80°F, clear skies"
    }
    # .get(key, default) means: look up city, and if it's not in the
    # dictionary, return this fallback string instead of crashing.
    # This is why asking about a city like "London" won't error —
    # it'll just honestly say no data is available.
    return fake_weather_data.get(city, f"No weather data available for {city}")


def get_time(city):
    fake_time_data = {
        "Chicago": "2:15 PM CDT",
        "New York": "3:15 PM EDT",
        "Los Angeles": "12:15 PM PDT"
    }
    return fake_time_data.get(city, f"No time data available for {city}")


# --- The tool menu Claude sees ---
# Claude never sees get_weather() or get_time() directly — it only ever
# sees this description. This is the ONLY information Claude has to decide
# whether, when, and how to call each tool. Name + description + expected
# input shape. Nothing about what's inside the function.
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


# --- Dispatcher ---
# This is the bridge between "Claude asked for this, by name, in JSON"
# and "here's the real Python function that actually does it."
# Claude can only ever produce text (including JSON) — it cannot execute
# get_weather() itself. This function is what actually runs it.
def run_tool(name, tool_input):
    if name == "get_weather":
        return get_weather(tool_input["city"])
    elif name == "get_time":
        return get_time(tool_input["city"])
    else:
        return f"Unknown tool: {name}"


# --- The ReAct loop ---
# messages holds the FULL conversation history. Claude has zero memory
# between API calls — every single client.messages.create() call must
# resend everything so far, or Claude has no idea what already happened.
# This list is what grows, turn by turn, as the loop runs.
messages = [
    {"role": "user", "content": "What's the weather and time in London right now?"}
]

# while True doesn't mean "forever" — it means "as many rounds as it
# actually takes." You don't know in advance whether Claude will need
# 1 tool call or 5, so you let response.stop_reason decide when to stop
# instead of hardcoding a number of rounds.
while True:
    # Send the ENTIRE history so far, plus the tool menu, to Claude.
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        tools=tools,
        messages=messages
    )

    # Add Claude's response to the conversation. If Claude asked for tools,
    # this turn includes those tool_use blocks — they get logged into
    # history just like any other assistant turn.
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason != "tool_use":
        # stop_reason is NOT "tool_use" — Claude decided it has everything
        # it needs and wrote a plain final answer instead of a tool request.
        # This is the loop's exit condition.
        final_text = next(block.text for block in response.content if block.type == "text")
        print("\n--- Final answer ---")
        print(final_text)
        break

    # stop_reason WAS "tool_use" — Claude wants to act before it can finish.
    # response.content can contain MULTIPLE tool_use blocks in one turn
    # (e.g. asking for weather AND time at once), so loop over all of them.
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            print(f"\n--- Claude is calling: {block.name} with {block.input} ---")
            result = run_tool(block.name, block.input)
            print(f"--- Result: {result} ---")
            # block.id (tool_use_id) is what lets Claude match this specific
            # result back to the specific call it answers — this matters
            # most when there were multiple tool calls in the same turn.
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result
            })

    # Feed all tool results back as the NEXT user turn. This is the
    # "Observe" step — Claude will read this on the next trip through
    # the loop and decide whether it now has enough to answer, or needs
    # to call a tool again.
    messages.append({"role": "user", "content": tool_results})