'''
JAX-based evaluator for SSA-generated XLA IR.

This evaluator parses the IR strings produced by `Expr.code` in SSA/Core.lean
and evaluates them using JAX operations, matching the Lean `DirectImpl` semantics.

Example::

    from python.eval import evaluate
    import jax.numpy as jnp

    ir = """%0 = const int [12] 1;
    %1 = const int [12] 0;
    %2 = where; $0, %0, %1
    %3 = cumsum; %2
    %4 = where; $0, %3, %1
    return %4"""

    x = jnp.array([0, 5, 0, 3, 0, 0, 7, 0, 1, 0, 0, 2], dtype=jnp.int32)
    result = evaluate(ir, x)

Extending the evaluator with a new operation::

    from python.eval import register_op
    import jax.numpy as jnp

    @register_op("my_op")
    def _eval_my_op(op_str, vals, lib_refs, libs, parent_args):
        x, y = vals
        return jnp.do_something(x, y)
'''

import re
from typing import List, Tuple, Dict, Callable, Any
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Operation registry
# ---------------------------------------------------------------------------

_OP_REGISTRY: Dict[str, Callable] = {}


def register_op(name: str):
    """Decorator to register an operation handler by its first word."""
    def decorator(fn: Callable):
        _OP_REGISTRY[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# IR parser
# ---------------------------------------------------------------------------

def parse_ir(code: str) -> Tuple[List[Tuple[str, str]], List[List[Tuple[str, str]]]]:
    """
    Parse an IR string into (main_body, libs).

    main_body: list of (lhs, rhs) instruction strings + return line
    libs: list of library bodies, where lib[i] is the body for &i
    """
    code = code.strip()
    sections = re.split(r"\n\n+", code)
    sections = [s.strip() for s in sections if s.strip()]

    if not sections:
        raise ValueError("Empty IR code")

    main_body = _parse_body(sections[0])
    libs = [_parse_body(s) for s in sections[1:]]
    return main_body, libs


def _parse_body(section: str) -> List[Tuple[str, str]]:
    """Parse a body section into list of (lhs, rhs) pairs."""
    lines = section.strip().split("\n")
    body = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("&"):
            continue
        if line.startswith("return"):
            body.append(("return", line[len("return "):].strip()))
        elif "=" in line:
            lhs, rhs = line.split("=", 1)
            body.append((lhs.strip(), rhs.strip()))
        else:
            raise ValueError(f"Unexpected line in IR body: {line!r}")
    return body


def _parse_args(arg_str: str) -> List[str]:
    """Split a comma-separated argument string, handling both ', ' and ','."""
    if not arg_str:
        return []
    return [p.strip() for p in arg_str.split(",")]


def _lib_input_count(lib_body: List[Tuple[str, str]]) -> int:
    """Count the number of inputs a library function expects (max $N + 1)."""
    max_arg = -1
    for lhs, rhs in lib_body:
        if lhs == "return":
            continue
        if ";" not in rhs:
            continue
        _, arg_str = rhs.split(";", 1)
        for a in _parse_args(arg_str):
            if a.startswith("$"):
                max_arg = max(max_arg, int(a[1:]))
    return max_arg + 1


def _lib_output_count(lib_body: List[Tuple[str, str]]) -> int:
    """Count the number of outputs a library function returns."""
    for lhs, rhs in lib_body:
        if lhs == "return":
            return len(_parse_args(rhs))
    raise ValueError("Library body has no return statement")


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

def eval_body(
    body: List[Tuple[str, str]],
    libs: List[List[Tuple[str, str]]],
    args: List[jnp.ndarray],
    vars_dict: Dict[str, jnp.ndarray] = None,
) -> List[jnp.ndarray]:
    """Evaluate a body (main or library) given input args and libs."""
    if vars_dict is None:
        vars_dict = {}
    else:
        vars_dict = dict(vars_dict)

    for lhs, rhs in body:
        if lhs == "return":
            ret_names = _parse_args(rhs)
            return [vars_dict[r] for r in ret_names]

        out_names = _parse_args(lhs.replace("%", ""))
        out_names = [f"%{n}" for n in out_names]

        if ";" in rhs:
            op_str, arg_str = rhs.split(";", 1)
            op_str = op_str.strip()
            raw_args = _parse_args(arg_str)
        else:
            op_str = rhs.strip()
            raw_args = []

        def resolve(a: str):
            if a.startswith("$"):
                return args[int(a[1:])]
            if a.startswith("%"):
                return vars_dict[a]
            if a.startswith("@"):
                return ("lib", int(a[1:]))
            raise ValueError(f"Unknown argument format: {a!r}")

        resolved = [resolve(a) for a in raw_args]

        val_list = [r for r in resolved if not (isinstance(r, tuple) and r[0] == "lib")]
        lib_ref_list = [r[1] for r in resolved if isinstance(r, tuple) and r[0] == "lib"]

        # Dispatch via registry (lookup by first word of op string)
        handler = _OP_REGISTRY.get(op_str.split()[0])

        if handler is None:
            raise ValueError(f"Unknown or unimplemented operation: {op_str!r}")

        result = handler(op_str, val_list, lib_ref_list, libs, args)

        if len(out_names) == 1:
            vars_dict[out_names[0]] = result
        else:
            if not isinstance(result, (list, tuple)):
                raise ValueError(f"Op {op_str!r} returned single value but expected {len(out_names)}")
            for name, val in zip(out_names, result):
                vars_dict[name] = val

    raise ValueError("Body has no return statement")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(code: str, *args: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate SSA-generated XLA IR code with the given input arrays.

    Args:
        code: The IR string generated by `Expr.code`.
        *args: Input arrays corresponding to $0, $1, ...

    Returns:
        The primary output array (first return value).
    """
    body, libs = parse_ir(code)
    results = eval_body(body, libs, list(args))
    if len(results) == 1:
        return results[0]
    return results


def evaluate_multi(code: str, *args: jnp.ndarray) -> List[jnp.ndarray]:
    """Evaluate and return all outputs as a list."""
    body, libs = parse_ir(code)
    return eval_body(body, libs, list(args))


# Import ops to populate the registry
import ops
