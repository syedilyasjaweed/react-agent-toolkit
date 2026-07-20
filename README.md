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
| `step3_agent_loop.py` | Same loop shape as step1, but `run_tool()` now calls the real function from step2 — real tool + real agent decision-making. |

## Testing results so far

| JD tested | Behavior observed |
|---|---|
| Oracle PL/SQL role | Weak first search (top score 0.49) → Claude reworded the query and retried on its own → honestly flagged the Oracle gap instead of overselling MySQL as a match. |
| Entry-level AI Engineer role | 3 separate tool calls across distinct skill areas (systems/deployment, cloud platforms, cross-functional collaboration), scores up to 0.57 → confident, well-supported "yes." |
| Pharmacy Technician role (deliberately out-of-domain) | Still called the tool once (checking is better than assuming), got uniformly low scores (0.27–0.30), and used that as evidence to recommend against applying. |
| Data Engineer role at a pharmacy company (partially out-of-domain) | Testing whether Claude separates "technical fit" from "industry fit" rather than collapsing into one verdict. |

**Key insight:** the interesting signal isn't *whether* Claude calls the tool — it almost always will, since checking is cheap. It's *how many times* it calls it, and whether its final confidence actually matches what the retrieval scores support.

## Debugging notes

- A `ModuleNotFoundError` on `from step2_real_tool import search_resume_bullets` turned out to be a leading space in the actual filename (`" step2_real_tool.py"`) — invisible at a glance, fatal to Python's exact-match import. Fixed by renaming the file.

## Roadmap

- [x] Step 1: fake-tool ReAct loop mechanics
- [x] Step 2: real `search_resume_bullets()` wired to Pinecone/Voyage AI, tested standalone
- [x] Step 3: real tool wired into the agent loop
- [x] Accept pasted job descriptions via `input()` in `step3_agent_loop.py`
- [x] Test a partially out-of-domain JD (Data Engineer @ pharmacy company)
- [ ] Add **company research** tool (Phase 2 of `ai-job-assistant`) to the tools list
- [ ] Add **cover letter generation** tool (Phase 3 of `ai-job-assistant`) to the tools list
- [ ] Multi-tool system prompt: one system prompt, Claude picks whichever tools fit — including multiple in one turn

## Related project

- [`ai-job-assistant`](https://github.com/syedilyasjaweed/ai-job-assistant) — the underlying 3-phase RAG pipeline (retrieval → company research → cover letter generation) this project turns into an autonomous agent.
