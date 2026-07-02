# Approach Document — SHL Conversational Assessment Recommender

## Architecture Overview

The system is a stateless FastAPI service with two endpoints: `GET /health` and `POST /chat`. Each `/chat` call receives the full conversation history and returns the next agent reply plus, when appropriate, a structured shortlist of recommendations.

**Core pipeline per request:**
1. **Guardrail checks** — Detect prompt injection and off-topic queries via regex patterns before calling the LLM.
2. **Requirement extraction (LLM Pass 1)** — Gemini 2.0 Flash analyzes the full conversation and extracts structured requirements (role, seniority, skills, test type preferences) as JSON.
3. **Semantic retrieval** — A FAISS index over 300+ SHL assessments (embedded with all-MiniLM-L6-v2) retrieves the top candidates. Optional metadata filters (job level, test type) narrow results.
4. **Response generation (LLM Pass 2)** — Given the retrieved assessments and extracted requirements, Gemini produces the conversational reply and selects the final shortlist.
5. **Validation** — Every recommended name and URL is verified against the loaded catalog before being returned.

## Design Choices

**Two-pass LLM approach.** Separating extraction from generation prevents hallucination: the LLM can only recommend from the catalog items it receives in context. This also decouples "understanding what the user wants" from "finding the right assessments."

**Sentence-Transformers + FAISS for retrieval.** With ~300 catalog items, a lightweight in-memory FAISS index provides sub-millisecond semantic search. The all-MiniLM-L6-v2 model handles natural language queries like "someone who handles pressure" better than keyword search. Post-retrieval metadata filters (job level, test category) add precision.

**Stateless API.** Every call re-analyzes the full conversation. This simplifies deployment (no session store) and matches the evaluator's design. The tradeoff is repeated LLM calls, but with ≤8 turns and Gemini Flash, latency stays under 10s.

**Guardrails at two levels.** Fast regex guards catch obvious injection and off-topic patterns before the LLM runs. The extraction prompt also outputs `is_off_topic`, providing a second LLM-powered check for ambiguous cases.

## Prompt Design

The system prompt enforces strict scope (SHL assessments only, no invented URLs). The extraction prompt uses a structured JSON schema so requirements are always machine-parseable. The recommendation prompt receives catalog data as context and is instructed to select only from that context. JSON response mode in Gemini ensures parseable outputs.

## What Didn't Work

- **Single-pass approach**: Early versions used one LLM call to extract requirements AND generate recommendations. This led to hallucinated assessment names. The two-pass design eliminated this.
- **Pure keyword matching**: TF-IDF over assessment descriptions missed semantic matches (e.g., "communication skills" → OPQ personality assessment). Sentence-Transformers solved this.
- **Overly strict off-topic detection**: Initial regex patterns blocked valid queries containing words like "legal" (e.g., "legal department hiring"). I relaxed regex guards and let the LLM handle ambiguous cases.

## Evaluation Approach

- **Schema compliance**: Unit tests verify every response matches the required JSON schema.
- **Guardrail tests**: Unit tests for injection detection, off-topic detection, turn limits, and catalog validation.
- **Recall@10 measurement**: Replay harness runs the 10 public conversation traces and computes mean Recall@10.
- **Behavior probes**: Manual tests for vague-query clarification, mid-conversation refinement, comparison grounding, and off-topic refusal.

## AI Tools Used

I used Google Gemini (via Antigravity) for AI-assisted code generation and iteration. All code was reviewed and understood before submission. Architecture and prompt design decisions are my own.
