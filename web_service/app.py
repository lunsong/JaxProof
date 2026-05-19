"""Flask web service for the SSA verification framework."""

import json
import os
import sys

# Ensure project root and python/ are importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "python"))

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from llm_client import LLMClient
from lean_runner import iterative_fix_and_build, iterative_fix_and_build_stream, _strip_markdown

# Lazy import of JAX evaluator (heavy dependency)
_evaluator = None

def _get_evaluator():
    global _evaluator
    if _evaluator is None:
        try:
            import eval as _eval_mod
            import ops  # noqa: F401 — side-effect: registers ops
            _evaluator = _eval_mod
        except Exception as e:
            raise RuntimeError(f"JAX evaluator not available: {e}")
    return _evaluator

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Initialize LLM client (raises on missing config)
try:
    llm = LLMClient()
except RuntimeError as e:
    print(f"WARNING: LLM client not initialized: {e}")
    llm = None


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


@app.route("/")
def index():
    """Serve the main frontend page."""
    return render_template("index.html")


@app.route("/api/formalize", methods=["POST"])
def api_formalize():
    """Step 1: Formalize natural language into a Lean theorem statement."""
    if llm is None:
        return jsonify({"error": "LLM client not configured. Set OPENAI_API_KEY."}), 500

    data = request.get_json(force=True)
    spec = data.get("spec", "").strip()
    if not spec:
        return jsonify({"error": "Empty specification."}), 400

    try:
        theorem_text = ""
        reasoning_text = ""
        for role, chunk in llm.formalize_stream(spec):
            if role == "reasoning":
                reasoning_text += chunk
            else:
                theorem_text += chunk
        theorem_text = _strip_markdown(theorem_text)
        return jsonify({
            "success": True,
            "theorem": theorem_text,
            "reasoning": reasoning_text,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/formalize-stream", methods=["POST"])
def api_formalize_stream():
    """Step 1 (streaming): Formalize NL spec with real-time CoT display."""
    if llm is None:
        return jsonify({"error": "LLM client not configured. Set OPENAI_API_KEY."}), 500

    data = request.get_json(force=True)
    spec = data.get("spec", "").strip()
    if not spec:
        return jsonify({"error": "Empty specification."}), 400

    def generate():
        try:
            theorem_text = ""
            reasoning_text = ""
            for role, chunk in llm.formalize_stream(spec):
                if role == "reasoning":
                    reasoning_text += chunk
                    yield _sse({"type": "reasoning", "chunk": chunk})
                else:
                    theorem_text += chunk
                    yield _sse({"type": "content", "chunk": chunk})
            theorem_text = _strip_markdown(theorem_text)
            yield _sse({
                "type": "done",
                "theorem": theorem_text,
                "reasoning": reasoning_text,
            })
        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/api/implement", methods=["POST"])
def api_implement():
    """Step 2: Generate SSA implementation + proof from theorem statement."""
    if llm is None:
        return jsonify({"error": "LLM client not configured. Set OPENAI_API_KEY."}), 500

    data = request.get_json(force=True)
    theorem_text = data.get("theorem", "").strip()
    auto_fix = data.get("auto_fix", True)
    max_fix_iters = data.get("max_fix_iterations", 3)

    if not theorem_text:
        return jsonify({"error": "Empty theorem text."}), 400

    try:
        lean_code = ""
        reasoning_text = ""
        for role, chunk in llm.implement_stream(theorem_text):
            if role == "reasoning":
                reasoning_text += chunk
            else:
                lean_code += chunk
        lean_code = _strip_markdown(lean_code)
    except Exception as e:
        return jsonify({"error": f"LLM implementation failed: {e}"}), 500

    if not auto_fix:
        return jsonify({
            "success": True,
            "lean_code": lean_code,
            "reasoning": reasoning_text,
            "verified": False,
        })

    # Try to build and fix
    try:
        success, msg, ir = iterative_fix_and_build(
            lean_code,
            llm,
            max_iterations=max_fix_iters,
        )
        return jsonify({
            "success": success,
            "lean_code": lean_code,
            "reasoning": reasoning_text,
            "message": msg,
            "ir": ir,
            "verified": success,
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "lean_code": lean_code,
            "reasoning": reasoning_text,
            "error": f"Build/fix failed: {e}",
            "verified": False,
        }), 500


@app.route("/api/implement-stream", methods=["POST"])
def api_implement_stream():
    """Step 2 (streaming): Generate SSA impl + proof with real-time CoT, then build."""
    if llm is None:
        return jsonify({"error": "LLM client not configured. Set OPENAI_API_KEY."}), 500

    data = request.get_json(force=True)
    theorem_text = data.get("theorem", "").strip()
    auto_fix = data.get("auto_fix", True)
    max_fix_iters = data.get("max_fix_iterations", 3)

    if not theorem_text:
        return jsonify({"error": "Empty theorem text."}), 400

    def generate():
        lean_code = ""
        reasoning_text = ""
        try:
            for role, chunk in llm.implement_stream(theorem_text):
                if role == "reasoning":
                    reasoning_text += chunk
                    yield _sse({"type": "reasoning", "chunk": chunk})
                else:
                    lean_code += chunk
                    yield _sse({"type": "content", "chunk": chunk})
            lean_code = _strip_markdown(lean_code)
        except Exception as e:
            yield _sse({"type": "error", "message": f"LLM generation failed: {e}"})
            return

        if not auto_fix:
            yield _sse({
                "type": "done",
                "lean_code": lean_code,
                "reasoning": reasoning_text,
                "verified": False,
            })
            return

        # Build phase
        yield _sse({"type": "phase", "message": "Compiling with Lean..."})
        try:
            final_success = False
            final_msg = ""
            final_ir = None
            for event in iterative_fix_and_build_stream(lean_code, llm, max_iterations=max_fix_iters):
                if event["type"] == "build_result":
                    if event["success"]:
                        final_success = True
                        final_msg = event["message"]
                        final_ir = event["ir"]
                        break
                elif event["type"] == "fix_start":
                    yield _sse({"type": "phase", "message": f"Build failed. Fixing (attempt {event['iteration']}/{max_fix_iters})..."})
                elif event["type"] == "fix_reasoning":
                    reasoning_text += event["chunk"]
                    yield _sse({"type": "reasoning", "chunk": event["chunk"]})
                elif event["type"] == "fix_content":
                    lean_code += event["chunk"]
                    yield _sse({"type": "content", "chunk": event["chunk"]})
                elif event["type"] == "fix_done":
                    lean_code = event["lean_code"]
                elif event["type"] == "failed":
                    final_msg = event["message"]
                    break

            yield _sse({
                "type": "done",
                "lean_code": lean_code,
                "reasoning": reasoning_text,
                "verified": final_success,
                "message": final_msg,
                "ir": final_ir,
            })
        except Exception as e:
            yield _sse({
                "type": "done",
                "lean_code": lean_code,
                "reasoning": reasoning_text,
                "verified": False,
                "error": str(e),
            })

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/api/verify", methods=["POST"])
def api_verify():
    """Verify existing Lean code (can be used to re-verify after user edits)."""
    data = request.get_json(force=True)
    lean_code = data.get("lean_code", "").strip()
    auto_fix = data.get("auto_fix", True)
    max_fix_iters = data.get("max_fix_iterations", 3)

    if not lean_code:
        return jsonify({"error": "Empty Lean code."}), 400

    if auto_fix and llm is None:
        return jsonify({"error": "LLM client not configured. Set OPENAI_API_KEY to use auto-fix."}), 500

    try:
        if auto_fix:
            success, msg, ir = iterative_fix_and_build(
                lean_code,
                llm,
                max_iterations=max_fix_iters,
            )
        else:
            from lean_runner import run_build
            success, msg, ir = run_build(lean_code)
        return jsonify({
            "success": success,
            "message": msg,
            "ir": ir,
            "verified": success,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """Evaluate generated IR with user-provided concrete inputs."""
    data = request.get_json(force=True)
    ir_code = data.get("ir", "").strip()
    args_spec = data.get("args", [])

    if not ir_code:
        return jsonify({"error": "Empty IR code."}), 400

    try:
        evaluator = _get_evaluator()
        import jax.numpy as jnp

        jax_arrays = []
        for spec in args_spec:
            shape = tuple(spec.get("shape", []))
            dtype_str = spec.get("dtype", "float")
            flat_data = spec.get("data", [])

            dtype = jnp.float32 if dtype_str == "float" else jnp.int32
            arr = jnp.array(flat_data, dtype=dtype).reshape(shape)
            jax_arrays.append(arr)

        result = evaluator.evaluate(ir_code, *jax_arrays)

        # Convert JAX array to nested Python list for JSON serialization
        result_list = jnp.asarray(result).tolist()
        result_shape = list(jnp.asarray(result).shape)
        result_dtype = "float" if jnp.asarray(result).dtype.kind == "f" else "int"

        return jsonify({
            "success": True,
            "result": {
                "data": result_list,
                "shape": result_shape,
                "dtype": result_dtype,
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def api_health():
    eval_ok = False
    try:
        _get_evaluator()
        eval_ok = True
    except Exception:
        pass
    return jsonify({
        "status": "ok",
        "llm_configured": llm is not None,
        "evaluator_ready": eval_ok,
    })


if __name__ == "__main__":
    # Run in threaded mode so one slow build doesn't block health checks
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=True)
