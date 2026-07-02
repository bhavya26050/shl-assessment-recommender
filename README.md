# SHL Conversational Assessment Recommender

A conversational AI agent that helps hiring managers and recruiters find the right SHL assessments through dialogue.

## Features

- **Clarification**: Asks follow-up questions when requests are vague
- **Recommendation**: Suggests 1-10 SHL assessments with catalog URLs
- **Refinement**: Updates recommendations when constraints change
- **Comparison**: Provides grounded comparisons between assessments
- **Guardrails**: Stays in scope, refuses off-topic and injection attempts

## Tech Stack

- **FastAPI** — REST API framework
- **Google Gemini 2.0 Flash** — LLM for conversation and reasoning
- **Lightweight hashed retrieval** — Default retrieval backend for low-memory deployment
- **Optional Sentence-Transformers** — Can be enabled locally with `USE_SENTENCE_TRANSFORMERS=true`

## Setup

1. Clone the repository
2. Create a `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
  To enable the heavier semantic backend locally, install `sentence-transformers` and `faiss-cpu`, then set `USE_SENTENCE_TRANSFORMERS=true`.
4. Run the server:
   ```bash
   py -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## API Endpoints

### GET /health
Returns service status.
```json
{"status": "ok"}
```

### POST /chat
Stateless chat endpoint. Send the full conversation history.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I need to assess a Java developer"}
  ]
}
```

**Response:**
```json
{
  "reply": "Sure! What seniority level are you looking for?",
  "recommendations": [],
  "end_of_conversation": false
}
```

## Docker

```bash
docker build -t shl-recommender .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key shl-recommender
```

## Testing

```bash
pytest tests/
```
