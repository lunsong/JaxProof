# SSA Verified Code Generator — Web Service

A web interface for automatically formalizing natural language specifications,
generating SSA (Static Single Assignment) implementations, and proving their
correctness in Lean 4.

## Workflow

1. **Specification** — User describes a function in natural language.
2. **Formalization** — LLM generates a Lean theorem statement.
3. **Review** — User can edit the formalization in a code editor.
4. **Implementation & Verification** — LLM writes the SSA program and proof,
   then `lake build Tests` verifies it.
5. **IR Output** — If verification succeeds, the generated XLA-like IR is displayed.

## Setup

```bash
# 1. Create virtual environment (already done)
python3 -m venv web_service/venv

# 2. Install dependencies
web_service/venv/bin/pip install flask flask-cors openai

# 3. Set your LLM API key
export OPENAI_API_KEY="sk-..."
# Optional: use a different base URL or model
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o"
```

## Running

```bash
cd web_service
source venv/bin/activate
python app.py
```

The service will be available at `http://localhost:5000`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend UI |
| `/api/health` | GET | Health check |
| `/api/formalize` | POST | NL spec → Lean theorem |
| `/api/implement` | POST | Theorem → SSA impl + proof + build |
| `/api/verify` | POST | Lean code → build + IR (no LLM needed if auto_fix=false) |

### Example API Usage

```bash
# Formalize
curl -X POST http://localhost:5000/api/formalize \
  -H "Content-Type: application/json" \
  -d '{"spec": "Compute the sum of squares of a vector"}'

# Implement and verify
curl -X POST http://localhost:5000/api/implement \
  -H "Content-Type: application/json" \
  -d '{"theorem": "import SSA\n\ndef sum_sq_spec...", "auto_fix": true}'

# Verify existing code
curl -X POST http://localhost:5000/api/verify \
  -H "Content-Type: application/json" \
  -d '{"lean_code": "import SSA\n...", "auto_fix": false}'
```

## Architecture

- **`app.py`** — Flask application with API endpoints.
- **`llm_client.py`** — Wrapper around OpenAI-compatible chat completion APIs.
- **`lean_runner.py`** — Writes temporary `.lean` files, runs `lake build Tests`,
  extracts IR from build output, and cleans up. Includes an iterative fix loop
  that asks the LLM to correct compilation errors.
- **`prompts.py`** — System prompts and few-shot examples for the LLM.
- **`templates/index.html`** — Single-page frontend.
- **`static/style.css`** — Minimal responsive styling.

## Notes

- The service modifies `Tests.lean` and `Tests/*.lean` temporarily during builds.
  A lock ensures only one build runs at a time. Cleanup happens automatically.
- On startup, any leftover `Tests/gen_*.lean` files from previous crashes are removed.
- Build timeout is 300 seconds.
- LLM-generated proofs may fail; the auto-fix loop retries up to 3 times by default.
