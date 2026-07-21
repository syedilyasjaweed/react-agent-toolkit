"""
Step 3: Wire search_resume_bullets(), research_company(), and the cover
letter generator into a real Claude tool-calling ReAct loop.

This file still has the exact same SHAPE as step1_tool_basics.py — same
kind of tools list, same run_tool dispatcher, same while-True loop. What's
changed since the first version: the tools list now has three real tools
instead of one, and the system prompt gives Claude judgment over when to
use each — including chaining multiple tools in a single turn.

Flow: Reason (Claude decides) -> Act (we run the tool it picked) ->
Observe (feed the result back as tool_result) -> loop -> Final answer.
"""

import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv
import anthropic

# --- Cross-project import setup ---
# ai-job-assistant is a sibling project folder, not a package inside
# react-agent-toolkit. Adding it to sys.path lets us import its real
# Phase 2 / Phase 3 functions directly instead of duplicating them here.
AI_JOB_ASSISTANT_PATH = Path.home() / "Developer" / "ai-job-assistant"
sys.path.insert(0, str(AI_JOB_ASSISTANT_PATH))

# search_resume_bullets is local to this project (step2_real_tool.py).
from step2_real_tool import search_resume_bullets

# research_company and the cover-letter pipeline pieces live in
# ai-job-assistant. phase3_cover_letter already imports research_company
# internally, but we also import it directly here so it can be exposed
# as its OWN standalone tool, separate from full letter generation.
from phase2_company_research import research_company
from phase3_cover_letter import (
    get_matched_bullets,
    get_company_profile,
    generate_cover_letter,
    save_cover_letter,
)

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# The system prompt shapes HOW Claude reasons before it ever sees the
# job description. With three tools now on the menu, this is also where
# Claude gets judgment about WHICH ones to reach for, and when it's fine
# to use more than one in a single turn.
SYSTEM_PROMPT = (
    "You are Syed's AI job-application assistant. You have three tools available:\n\n"
    "- search_resume_bullets: check resume evidence for specific skills or requirements.\n"
    "- research_company: get a live, structured profile of a company (mission, recent "
    "news, tech stack, culture).\n"
    "- generate_cover_letter: produce and save a tailored cover letter for a specific "
    "role at a specific company.\n\n"
    "Decide for yourself which tools, if any, actually help you answer well. You can "
    "call more than one in a single turn when that's useful — for example, checking "
    "resume fit and researching the company at the same time. Only call "
    "generate_cover_letter when the user has actually asked for a cover letter or "
    "clearly wants one produced; it's a heavier, user-facing action that writes a file, "
    "not something to run just to explore. When assessing job fit, lean on "
    "search_resume_bullets and be honest about gaps rather than overselling a match. "
    "Only call a tool when it will genuinely improve your answer."
)

# --- 1. Tool schemas Claude sees ---
# Claude never sees the function bodies behind these — not search_resume_bullets(),
# not research_company(), not generate_cover_letter(). It only ever sees these
# descriptions. The description is the ONLY thing Claude uses to decide when
# and how to call each one.
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
    },
    {
        "name": "research_company",
        "description": (
            "Researches a company using live web search and returns a structured profile: "
            "mission/values, recent news, tech stack, and culture notes. Use this when you "
            "need concrete, current information about a specific company — e.g. before "
            "assessing culture fit or before drafting a cover letter. Results are cached "
            "for 7 days, so repeated calls for the same company are fast."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "The exact name of the company to research.",
                },
                "force_refresh": {
                    "type": "boolean",
                    "description": "Set true to bypass the cache and re-research even if a recent profile exists. Defaults to false.",
                },
            },
            "required": ["company_name"],
        },
    },
    {
        "name": "generate_cover_letter",
        "description": (
            "Generates a tailored, ready-to-send cover letter (under 400 words) for a "
            "specific job application, and saves it to disk. This tool internally re-runs "
            "resume matching and company research on its own, so call it directly once you "
            "have the company name, role title, and job description — you don't need to "
            "call search_resume_bullets or research_company first, though doing so can help "
            "you decide whether generating a letter is actually warranted."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company": {
                    "type": "string",
                    "description": "Target company name.",
                },
                "role": {
                    "type": "string",
                    "description": "Target job title / role.",
                },
                "jd_text": {
                    "type": "string",
                    "description": "The full job description text.",
                },
                "force_refresh": {
                    "type": "boolean",
                    "description": "Set true to bypass the company research cache. Defaults to false.",
                },
            },
            "required": ["company", "role", "jd_text"],
        },
    },
]


# --- 2. Dispatcher: run whichever tool Claude asked for ---
# Same job as before — the bridge between "Claude asked for this by name
# in JSON" and "here's the real function that executes it." Now with
# three real branches instead of one.
def run_tool(tool_name, tool_input):
    if tool_name == "search_resume_bullets":
        query = tool_input["query"]
        top_k = tool_input.get("top_k", 5)  # default to 5 if Claude doesn't specify
        return search_resume_bullets(query, top_k=top_k)

    elif tool_name == "research_company":
        company_name = tool_input["company_name"]
        force_refresh = tool_input.get("force_refresh", False)
        profile = research_company(company_name, force_refresh=force_refresh)
        return json.dumps(profile)

    elif tool_name == "generate_cover_letter":
        company = tool_input["company"]
        role = tool_input["role"]
        jd_text = tool_input["jd_text"]
        force_refresh = tool_input.get("force_refresh", False)

        # This mirrors what phase3_cover_letter.py's main() does, just
        # returned instead of printed, since it's now a tool call and
        # not a CLI entry point.
        bullets = get_matched_bullets(jd_text)
        profile = get_company_profile(company, force_refresh)
        letter = generate_cover_letter(company, role, jd_text, bullets, profile)
        output_path = save_cover_letter(company, role, letter)

        return json.dumps({
            "cover_letter": letter,
            "saved_to": str(output_path),
        })

    return f"Unknown tool: {tool_name}"


# --- 3. The ReAct loop, one turn at a time ---
# This is the same tool-calling loop as before, but it no longer owns
# `messages` itself — it receives the list from the caller and mutates
# it in place (appending assistant turns and tool results as it goes).
# That's what makes multi-turn possible: the caller can keep reusing the
# same growing list across several calls to this function, so Claude
# still has everything said earlier in the conversation.
def run_agent_turn(messages):
    while True:
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[Claude]: {block.text}")

        if response.stop_reason != "tool_use":
            # Claude's final answer for THIS turn. Append it to history
            # before returning, so the next user turn's API call includes
            # it — otherwise Claude would "forget" its own last answer.
            messages.append({"role": "assistant", "content": response.content})
            final_text = "".join(b.text for b in response.content if b.type == "text")
            print(f"\n=== FINAL ANSWER ===\n{final_text}")
            return final_text

        messages.append({"role": "assistant", "content": response.content})

        tool_result_blocks = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\n[Tool call] {block.name}({block.input})")
                result = run_tool(block.name, block.input)
                print(f"[Tool result]\n{result}")
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

        messages.append({"role": "user", "content": tool_result_blocks})


# --- 4. Multi-turn conversation loop ---
# Owns the one `messages` list for the whole conversation. Each pass:
# collect a new user message, append it, hand the list to run_agent_turn
# (which may call tools any number of times before answering), then loop
# back for the next message. The conversation only ends when you type
# QUIT — until then, Claude has the full history: earlier fit checks,
# company research, and its own prior answers.
def chat():
    print("Multi-turn job assistant. Paste a message, then type END on its")
    print("own line to send it. Type QUIT (instead of a message) to exit.\n")

    messages = []
    while True:
        print("--- Your turn ---")
        lines = []
        while True:
            line = input()
            stripped = line.strip()
            if stripped == "END":
                break
            if stripped == "QUIT":
                print("\nEnding conversation.")
                return
            lines.append(line)

        user_message = "\n".join(lines).strip()
        if not user_message:
            # Nothing typed before END — skip rather than send an empty turn.
            continue

        messages.append({"role": "user", "content": user_message})
        run_agent_turn(messages)


if __name__ == "__main__":
    chat()