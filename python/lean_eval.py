"""
Bridge from Lean declarations to the JAX evaluator.

Resolves a qualified declaration name like ``"Examples.offDiag"`` to the IR
string produced by ``Expr.code`` (by running a one-line script through the
same Lean toolchain that lean-lsp-mcp drives — ``lake build`` for the olean,
then ``lake env lean`` — but with no LSP session or extra dependencies),
then hands it to the registry-based evaluator in ``eval.py``.

Example::

    from lean_eval import eval
    import jax.numpy as jnp

    x = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
    eval("Examples.offDiag", x, params=[("n", 4)])

    # positional instantiation works too:
    # eval("Examples.SchNet.embed", Z, theta, params=[10, 20, 30])
"""

import re
import subprocess
import tempfile
from pathlib import Path
from functools import lru_cache
from typing import Union, Tuple, Any

from eval import evaluate

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# A parameter is either a literal (positional) or a ("name", value) pair.
Param = Union[int, str, Tuple[str, Any]]


def _render_param(p: Param) -> str:
    if isinstance(p, tuple):
        return f"({p[0]} := {p[1]})"
    return str(p)


@lru_cache(maxsize=None)
def _code_cached(module: str, decl: str, params: Tuple[Param, ...], root: str) -> str:
    app = "".join(" " + _render_param(p) for p in params)
    script = f"import {module}\n#eval IO.println ({decl}{app}).code\n"

    build = subprocess.run(
        ["lake", "build", module], cwd=root, capture_output=True, text=True
    )
    if build.returncode != 0:
        raise RuntimeError(
            f"`lake build {module}` failed:\n{build.stderr.strip()[-2000:]}"
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", prefix="jaxproof_eval_", delete=False
    ) as f:
        f.write(script)
        tmp = f.name
    try:
        proc = subprocess.run(
            ["lake", "env", "lean", tmp], cwd=root, capture_output=True, text=True
        )
    finally:
        Path(tmp).unlink(missing_ok=True)

    if proc.returncode != 0:
        hint = ""
        if not params and re.search(
            r"unsolved goals|metavariable|synthesize implicit", proc.stderr
        ):
            hint = (
                "\n(hint: the declaration may take implicit arguments — "
                "instantiate them with params=[(\"n\", 8), ...])"
            )
        raise RuntimeError(
            f"Lean failed to elaborate `{decl}{app}` from {module}:{hint}\n"
            f"{proc.stderr.strip()[-2000:]}"
        )
    return proc.stdout.strip()


def lean_code(
    ref: str,
    *params: Param,
    project_root: Union[str, Path] = PROJECT_ROOT,
) -> str:
    """
    Return the IR string for a Lean expression declaration.

    Args:
        ref: qualified name "Module.Path.decl" (last component = declaration).
        *params: instantiation arguments for the declaration; an int/str is
            applied positionally, a ("name", value) pair becomes `(name := value)`.
        project_root: repository root containing the lakefile.
    """
    module, _, decl = ref.rpartition(".")
    if not module or not decl:
        raise ValueError(
            f"Expected a qualified name like 'Examples.offDiag', got {ref!r}"
        )
    return _code_cached(module, decl, tuple(params), str(project_root))


def lean_eval(ref: str, *args, params: Tuple[Param, ...] = (), **kwargs):
    """
    Evaluate a Lean expression declaration on JAX arrays.

    `ref` is a qualified declaration name (e.g. "Examples.offDiag"); the
    declaration is instantiated with `params` (if any), its `.code` is
    fetched through the Lean toolchain (cached), and the IR is evaluated
    with `evaluate` on `args`. Extra keyword arguments are forwarded to
    `evaluate` (e.g. `intermediates=[...]`).
    """
    return evaluate(lean_code(ref, *params), *args, **kwargs)
