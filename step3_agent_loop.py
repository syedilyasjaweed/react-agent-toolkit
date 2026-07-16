"""
Step 3: Wire search_resume_bullets() into a real Claude tool-calling ReAct loop.

This file has the exact same SHAPE as step1_tool_basics.py — same tools
list, same run_tool dispatcher, same while-True loop. The only thing that
changed is WHAT the tool actually does: instead of get_weather/get_time
(fake dictionary lookups), it's search_resume_bullets — a real function
from step2_real_tool.py that calls Voyage AI and Pinecone.

Flow: Reason (Claude decides) -> Act (we run the tool it picked) ->
Observe (feed the result back as tool_result) -> loop -> Final answer.
"""

import os
from dotenv import load_dotenv
import anthropic

# This import is the actual wiring moment: step2_real_tool.py has no
# Claude API code in it at all — it's just a plain function. Here we
# pull that real function in so run_tool() can call it for real.
from step2_real_tool import search_resume_bullets

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# The system prompt shapes HOW Claude reasons before it ever sees the
# job description. It's what tells Claude "you're allowed to decide for
# yourself whether a tool call helps" and "don't call it if it won't."
# Without this nudge, Claude might call the tool reflexively every time,
# or never consider it at all.
SYSTEM_PROMPT = (
    "You are Syed's AI job-application assistant. When given a job description or a "
    "question about job fit, decide for yourself whether you need concrete evidence "
    "from Syed's resume to answer well. If so, call search_resume_bullets with a query "
    "that captures the role's core requirements. You can call it more than once with "
    "different queries if the job has distinct skill areas worth checking separately. "
    "Only call the tool when it will actually improve your answer."
)

# --- 1. Tool schema Claude sees ---
# Same idea as step1's tools list: Claude never sees search_resume_bullets()
# itself, or Voyage, or Pinecone. It only ever sees this description —
# name, what it's for, what input it expects. This description is the
# ONLY thing Claude uses to decide when and how to call it.
tools = [
    {
        "name": "search_resume_bullets",
        "description": (
            "Semantically searches Syed's resume bullets and returns the most relevant "
            "ones for a given query. Use this to find evidence of specific skills, tools, "
            "or experience relevant to a job description, role title, or requirement. "
            "The query is embedded and matched by meaning, not exact keywords."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Text to match against resume bullets — e.g. a job description, a requirement, or a skill/role name.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "How many top matching bullets to return. Defaults to 5.",
                },
            },
            "required": ["query"],
        },
    }
]


# --- 2. Dispatcher: run whichever tool Claude asked for ---
# Same job as step1's run_tool() — the bridge between "Claude asked for
# this by name in JSON" and "here's the real function that executes it."
# The only difference: this one calls a real function that hits real APIs,
# instead of a dictionary lookup.
def run_tool(tool_name, tool_input):
    if tool_name == "search_resume_bullets":
        query = tool_input["query"]
        top_k = tool_input.get("top_k", 5)  # default to 5 if Claude doesn't specify
        return search_resume_bullets(query, top_k=top_k)
    return f"Unknown tool: {tool_name}"


# --- 3. The ReAct loop ---
# Wrapped in a function this time (run_agent) so you can call it with
# different job descriptions without rewriting the loop each time.
# Same core idea as step1: messages holds the full history, because
# Claude has zero memory between API calls and needs the whole
# conversation resent every round.
def run_agent(user_message):
    messages = [{"role": "user", "content": user_message}]

    # while True = "as many rounds as it actually takes." Claude's own
    # stop_reason decides when to stop — not a hardcoded number of turns.
    while True:
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        # Print any reasoning text Claude produced before/alongside a tool
        # call — this is what makes Claude's decision-making visible while
        # you're learning, e.g. seeing it explain why it's searching again.
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[Claude]: {block.text}")

        if response.stop_reason != "tool_use":
            # Claude decided it has enough evidence (or needs none) and
            # wrote a final answer instead of requesting a tool. Exit here.
            final_text = "".join(b.text for b in response.content if b.type == "text")
            print(f"\n=== FINAL ANSWER ===\n{final_text}")
            return final_text

        # Claude wants to act — log its turn (including the tool_use
        # block(s)) into history before you go run anything.
        messages.append({"role": "assistant", "content": response.content})

        # response.content can hold MULTIPLE tool_use blocks in one turn
        # (e.g. two different searches at once), so handle all of them.
        tool_result_blocks = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\n[Tool call] {block.name}({block.input})")
                result = run_tool(block.name, block.input)
                print(f"[Tool result]\n{result}")
                # block.id ties this result back to the exact call it
                # answers — same purpose as tool_use_id in step1.
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

        # Feed results back as the next "user" turn — this is "Observe."
        # Claude reads this on the next loop iteration and decides whether
        # it now has enough, or needs to call the tool again with a
        # different query (exactly what happened with the Oracle PL/SQL
        # test — a weak first result led Claude to try a second search).
        messages.append({"role": "user", "content": tool_result_blocks})


if __name__ == "__main__":
    print("Paste the job description below.")
    print("When you're done, type END on its own line and press Enter:\n")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    job_description = "\n".join(lines)

    run_agent(job_description)