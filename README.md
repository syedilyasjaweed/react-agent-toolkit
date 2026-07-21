# react-agent-toolkit

A from-scratch implementation of the **ReAct (Reason → Act → Observe) loop** using the Claude API's tool-calling interface, built on top of the [`ai-job-assistant`](https://github.com/syedilyasjaweed/ai-job-assistant) RAG pipeline. This project is the "agentic" upgrade layer: instead of running the RAG pipeline as a fixed sequence of steps, Claude decides for itself — on each turn — whether it needs to call a tool, which one, and how many times.

## Why this exists

`ai-job-assistant` proved the retrieval pipeline works (Voyage AI embeddings + Pinecone search over resume bullets). `react-agent-toolkit` proves something different: that Claude can be handed a *menu* of tools and a job description, and autonomously reason about which tools to call, in what order, and how many times — rather than following a hardcoded script.

## Core concepts

- **Agent vs. chatbot** — An agent decides for itself, each turn, whether it needs a tool. The signal is Claude's `stop_reason` coming back as `"tool_use"` instead of plain text.
- **Claude can't execute code — it can only ask, in JSON** — A tool call is Claude producing something like `{"name": "search_resume_bullets", "input": {"query": "..."}}`. Claude never touches Voyage AI or Pinecone directly. `run_tool()` is the bridge between "Claude asked for this" and the real function that executes it.
- **The tool schema is a menu, not code** — Claude only ever sees a tool's `name`, `description`, and input shape — never the function body. The description is the sole basis for Claude's decision to call it. Better descriptions → better decisions.
- **Zero memory between API calls** — Every `client.messages.create()` call resends the full `messages` list. The loop works by appending both Claude's turns and tool results back into that list each round.
- **The ReAct loop** — Reason (Claude decides) → Act (`run_tool()` executes) → Observe (result returned as a `tool_result`) → loop back to Reason → repeat until `stop_reason != "tool_use"` → final answer. `while True` means "as many rounds as it actually takes," not "forever."
- **`tool_use_id` matters with simultaneous calls** — A single turn can contain multiple `tool_use` blocks. Each gets a unique ID so its `tool_result` maps back to the exact call it's answering.

## Project structure

| File | Purpose |
|---|---|
| `step1_tool_basics.py` | Fake tools (`get_weather`, `get_time`, hardcoded dicts), no real APIs. Learn the loop mechanics risk-free. |
| `step2_real_tool.py` | The real `search_resume_bullets()` function — Voyage AI embeds the query, Pinecone finds nearest resume-bullet vectors. Plain callable function, no Claude API call, no loop. Tested standalone. |
| `step3_agent_loop.py` | The full multi-turn agent. Wires three real tools into one ReAct loop: `search_resume_bullets` (step2), `research_company` (Phase 2 of `ai-job-assistant`, live web search), and `generate_cover_letter` (Phase 3, full matcher + research + Claude generation pipeline). Supports an ongoing conversation across multiple turns, not just a single one-shot message. |

## Multi-tool architecture

Three tools now sit on the same menu, and one system prompt gives Claude judgment over all of them — including calling more than one in a single turn:

- **`search_resume_bullets`** — checks resume evidence for specific skills/requirements (step2).
- **`research_company`** — live web search → structured profile (mission, recent news, tech stack, culture), cached for 7 days.
- **`generate_cover_letter`** — a heavier, user-facing action: internally re-runs resume matching + company research, generates a tailored letter, and saves it to disk. The system prompt explicitly tells Claude not to fire this one speculatively — only when the user has actually asked for a letter.

`run_tool()`'s dispatcher branches on all three; the ReAct loop itself didn't need to change at all to support this, since it already handled multiple `tool_use` blocks per turn.

## Multi-turn conversation support

`step3_agent_loop.py` now runs as an ongoing conversation (`chat()`) instead of a single one-shot message. `run_agent_turn(messages)` handles one turn's worth of tool-calling and mutates the shared `messages` list in place, so a follow-up like "yes, write me a cover letter for that" still has the full context — the JD, the resume scores, the company profile — from earlier in the same conversation. Type `END` to send a message, `QUIT` to exit.

## Testing results so far

| JD tested | Behavior observed |
|---|---|
| Oracle PL/SQL role | Weak first search (top score 0.49) → Claude reworded the query and retried on its own → honestly flagged the Oracle gap instead of overselling MySQL as a match. |
| Entry-level AI Engineer role | 3 separate tool calls across distinct skill areas (systems/deployment, cloud platforms, cross-functional collaboration), scores up to 0.57 → confident, well-supported "yes." |
| Pharmacy Technician role (deliberately out-of-domain) | Still called the tool once (checking is better than assuming), got uniformly low scores (0.27–0.30), and used that as evidence to recommend against applying. |
| Software Engineer (Gen AI) @ Virtusa | Real end-to-end run with all context wired in: called `search_resume_bullets` (scores 0.42–0.55) and `research_company` in the same turn, then gave an honest fit assessment — flagged missing Scrum/cloud/ML-library evidence as "preferred, not required" gaps rather than glossing over them. Correctly did *not* call `generate_cover_letter` unprompted, just offered to. |
| Early Career Software Engineer @ Wonderschool | Full multi-turn chain: fit check → asked for a cover letter in a follow-up turn → `generate_cover_letter` fired, using context carried over from the earlier turns. Surfaced a real bug in the process (see Debugging notes). |

**Key insight:** the interesting signal isn't *whether* Claude calls a tool — it almost always will, since checking is cheap. It's *how many times* it calls each one, whether it reaches for the right combination, and whether its final confidence actually matches what the retrieval scores and research support.

## Debugging notes

- A `ModuleNotFoundError` on `from step2_real_tool import search_resume_bullets` turned out to be a leading space in the actual filename (`" step2_real_tool.py"`) — invisible at a glance, fatal to Python's exact-match import. Fixed by renaming the file.
- **`max_tokens` truncation corrupting conversation history:** `generate_cover_letter`'s tool call has to reproduce the *entire* job description inside its JSON input. On a long JD, this pushed the response past `max_tokens=1024` and cut it off mid-`tool_use`. The original code treated any non-`tool_use` stop reason as a final answer and appended it to `messages` anyway — silently leaving a `tool_use` block with no matching `tool_result` baked permanently into that conversation's history. Every subsequent turn then failed with a 400 error (`tool_use ids were found without tool_result blocks`). Fixed two ways: raised `max_tokens` to 8192 for headroom, and added an explicit check that discards (rather than saves) any response that hit `max_tokens`, so a truncated turn can no longer corrupt history — it just prompts a retry.
- **Cost note:** `research_company`'s web search tool bills per search performed, separately from normal token costs, and can run several searches internally per call — this was the actual source of a ~24¢ single-call cost, not the base `messages.create()` token usage.

## Roadmap

- [x] Step 1: fake-tool ReAct loop mechanics
- [x] Step 2: real `search_resume_bullets()` wired to Pinecone/Voyage AI, tested standalone
- [x] Step 3: real tool wired into the agent loop
- [x] Accept pasted job descriptions via `input()` in `step3_agent_loop.py`
- [x] Test a partially out-of-domain JD (Data Engineer @ pharmacy company)
- [x] Add **company research** tool (Phase 2 of `ai-job-assistant`) to the tools list
- [x] Add **cover letter generation** tool (Phase 3 of `ai-job-assistant`) to the tools list
- [x] Multi-tool system prompt: one system prompt, Claude picks whichever tools fit — including multiple in one turn
- [x] Multi-turn conversation support (`chat()` loop, shared `messages` history)
- [x] Fix `max_tokens` truncation corrupting conversation history

**Project complete.** All planned steps are wired in and tested end-to-end: a fake-tool ReAct loop grew into a real, multi-turn agent with three working tools (resume search, company research, cover letter generation), verified against real job descriptions from Oracle PL/SQL roles to a full Virtusa and Wonderschool run.

## Related project

- [`ai-job-assistant`](https://github.com/syedilyasjaweed/ai-job-assistant) — the underlying 3-phase RAG pipeline (retrieval → company research → cover letter generation) this project turns into an autonomous agent.