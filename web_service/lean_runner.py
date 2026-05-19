"""Utilities for running Lean builds and extracting generated IR."""

import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Tuple

# Path to project root (parent of web_service/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def _sanitize_module_name(name: str) -> str:
    """Convert a string into a valid Lean module name."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _extract_ir(stdout: str) -> Optional[str]:
    """Extract IR code from `lake env lean` stdout."""
    lines = stdout.splitlines()
    ir_lines = []
    for line in lines:
        # Stop when we hit a Lean warning/error line
        if re.match(r"^/.*\.lean:\d+:\d+:", line):
            break
        # Skip leading blank lines
        if not ir_lines and not line.strip():
            continue
        ir_lines.append(line)
    # Trim trailing blank lines
    while ir_lines and not ir_lines[-1].strip():
        ir_lines.pop()
    if ir_lines:
        return "\n".join(ir_lines)
    return None


def _extract_error(stdout: str) -> str:
    """Extract a concise error message from `lake env lean` output."""
    lines = stdout.splitlines()
    # Look for error blocks
    relevant = []
    collecting = False
    for line in lines:
        if re.match(r"^/.*\.lean:\d+:\d+:\s*error", line):
            collecting = True
        if collecting:
            relevant.append(line)
            if len(relevant) > 40:
                break
    if relevant:
        return "\n".join(relevant[:40])
    # Fallback: return everything that looks like an error or warning
    return "\n".join(lines[-50:])


def run_build(
    lean_code: str,
    module_name: Optional[str] = None,
    cleanup: bool = True,
) -> Tuple[bool, str, Optional[str]]:
    """
    Write lean_code to a temp file and run `lake env lean` on it.
    Extracts IR from stdout and errors from stderr/stdout.

    Returns:
        (success: bool, message_or_error: str, ir_code: Optional[str])
    """
    if module_name is None:
        module_name = f"gen_{uuid.uuid4().hex[:8]}"
    module_name = _sanitize_module_name(module_name)

    # Write to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", prefix=f"ssa_{module_name}_", delete=False
    ) as f:
        f.write(lean_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["lake", "env", "lean", tmp_path],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout + result.stderr

        if result.returncode == 0:
            ir = _extract_ir(result.stdout)
            return True, "Build succeeded!", ir
        else:
            error = _extract_error(output)
            return False, error, None
    except subprocess.TimeoutExpired:
        return False, "Build timed out after 300 seconds.", None
    except Exception as e:
        return False, f"Build failed with exception: {e}", None
    finally:
        if cleanup:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def iterative_fix_and_build(
    lean_code: str,
    llm_client,
    module_name: Optional[str] = None,
    max_iterations: int = 3,
) -> Tuple[bool, str, Optional[str]]:
    """
    Attempt to build the code. If it fails, ask the LLM to fix it and retry.

    Returns:
        (success: bool, message: str, ir_code: Optional[str])
    """
    current_code = lean_code
    for iteration in range(max_iterations):
        success, msg, ir = run_build(current_code, module_name, cleanup=True)
        if success:
            return True, msg, ir
        if iteration == max_iterations - 1:
            return False, f"Failed after {max_iterations} attempts. Last error:\n{msg}", None

        # Ask LLM to fix
        fixed = llm_client.fix(current_code, msg)
        # Strip markdown if present
        fixed = _strip_markdown(fixed)
        current_code = fixed

    return False, "Unexpected exit from fix loop.", None


def iterative_fix_and_build_stream(
    lean_code: str,
    llm_client,
    module_name: Optional[str] = None,
    max_iterations: int = 3,
):
    """
    Streaming version of iterative_fix_and_build.
    Yields dict events so callers can stream progress to the client.

    Event types:
    - {"type": "build_result", "success": bool, "message": str, "ir": str|None}
    - {"type": "fix_start", "iteration": int, "error": str}
    - {"type": "fix_reasoning", "chunk": str}
    - {"type": "fix_content", "chunk": str}
    - {"type": "fix_done", "lean_code": str}
    - {"type": "failed", "message": str}
    """
    current_code = lean_code
    for iteration in range(max_iterations):
        success, msg, ir = run_build(current_code, module_name, cleanup=True)
        yield {"type": "build_result", "success": success, "message": msg, "ir": ir}
        if success:
            return
        if iteration == max_iterations - 1:
            yield {"type": "failed", "message": f"Failed after {max_iterations} attempts. Last error:\n{msg}"}
            return

        yield {"type": "fix_start", "iteration": iteration + 1, "error": msg}
        fixed = ""
        for role, chunk in llm_client.fix_stream(current_code, msg):
            if role == "reasoning":
                yield {"type": "fix_reasoning", "chunk": chunk}
            elif role == "content":
                fixed += chunk
                yield {"type": "fix_content", "chunk": chunk}
        fixed = _strip_markdown(fixed)
        current_code = fixed
        yield {"type": "fix_done", "lean_code": fixed}


def _strip_markdown(text: str) -> str:
    """Remove markdown code block wrappers if present."""
    text = text.strip()
    if text.startswith("```lean"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
