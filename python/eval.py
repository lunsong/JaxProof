'''
JAX-based evaluator for SSA-generated XLA IR.

This evaluator parses the IR strings produced by `Expr.code` in Soir/Core.lean
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
from typing import Dict, List, Tuple, Callable, Optional, Iterable, Any, Union
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


def registered_ops() -> List[str]:
    """Names of all registered operations."""
    return sorted(_OP_REGISTRY)


# ---------------------------------------------------------------------------
# IR parser
# ---------------------------------------------------------------------------

# A parsed body is a list of instructions. Each instruction is either
#   ("assign", [out_name, ...], op_str, [raw_arg, ...])
#   ("return", [ret_name, ...])
Instruction = Tuple


def parse_ir(code: str) -> Tuple[List[Instruction], List[List[Instruction]]]:
    """
    Parse an IR string into (main_body, libs).

    Bodies are lists of instructions; `libs[i]` is the body referenced by `@i`.
    """
    code = code.strip()
    sections = [s.strip() for s in re.split(r"\n\n+", code) if s.strip()]

    if not sections:
        raise ValueError("Empty IR code")

    main_body = _parse_body(sections[0])

    libs: List[List[Instruction]] = []
    for section in sections[1:]:
        header, _, rest = section.partition("\n")
        m = re.fullmatch(r"@(\d+):", header.strip())
        if not m:
            raise ValueError(f"Library section must start with '@<id>:', got {header!r}")
        idx = int(m.group(1))
        if idx != len(libs):
            raise ValueError(
                f"Library sections must appear in order (@0, @1, ...); "
                f"got @{idx} at position {len(libs)}"
            )
        libs.append(_parse_body(rest))
    return main_body, libs


def _parse_body(section: str) -> List[Instruction]:
    """Parse a body section into a list of instructions."""
    body: List[Instruction] = []
    for line in section.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("return"):
            body.append(("return", _parse_args(line[len("return"):].strip())))
        elif "=" in line:
            lhs, rhs = line.split("=", 1)
            out_names = [a.strip() for a in lhs.split(",")]
            for name in out_names:
                if not re.fullmatch(r"%\d+", name):
                    raise ValueError(f"Invalid assignment target {name!r} in line {line!r}")
            if ";" in rhs:
                op_str, arg_str = rhs.split(";", 1)
                raw_args = _parse_args(arg_str)
            else:
                op_str, raw_args = rhs.strip(), []
            body.append(("assign", out_names, op_str.strip(), raw_args))
        else:
            raise ValueError(f"Unexpected line in IR body: {line!r}")
    if not body or body[-1][0] != "return":
        raise ValueError("Body has no return statement")
    return body


def _parse_args(arg_str: str) -> List[str]:
    """Split a comma-separated argument string."""
    if not arg_str:
        return []
    return [p.strip() for p in arg_str.split(",")]


def _lib_output_count(lib_body: List[Instruction]) -> int:
    """Count the number of outputs a library function returns."""
    return len(lib_body[-1][1])  # the return instruction is verified last


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

def _resolve(a: str, args: List[jnp.ndarray], vars_dict: Dict[str, jnp.ndarray]):
    if a.startswith("$"):
        return args[int(a[1:])]
    if a.startswith("%"):
        try:
            return vars_dict[a]
        except KeyError:
            raise ValueError(f"Variable {a!r} used before assignment") from None
    if a.startswith("@"):
        return ("lib", int(a[1:]))
    raise ValueError(f"Unknown argument format: {a!r}")


def eval_body(
    body: List[Instruction],
    libs: List[List[Instruction]],
    args: List[jnp.ndarray],
    vars_dict: Optional[Dict[str, jnp.ndarray]] = None,
    intermediates: Optional[Dict[int, jnp.ndarray]] = None,
) -> List[jnp.ndarray]:
    """
    Evaluate a parsed body (main or library) given input args and libs.

    If `intermediates` is a dict, every assigned variable is recorded in it
    under its numeric id (e.g. `%3` -> `intermediates[3]`).
    """
    vars_dict = dict(vars_dict) if vars_dict is not None else {}

    for instr in body:
        if instr[0] == "return":
            # Return names may reference inputs directly (e.g. `return %0, $1`)
            # when an expression passes an argument through unchanged.
            return [_resolve(r, args, vars_dict) for r in instr[1]]

        _, out_names, op_str, raw_args = instr

        try:
            resolved = [_resolve(a, args, vars_dict) for a in raw_args]
        except ValueError as e:
            lhs = ",".join(out_names)
            raise ValueError(f"{e} [in `{lhs} = {op_str}; {', '.join(raw_args)}`]") from None

        val_list = [r for r in resolved if not (isinstance(r, tuple) and r[0] == "lib")]
        lib_ref_list = [r[1] for r in resolved if isinstance(r, tuple) and r[0] == "lib"]

        handler = _OP_REGISTRY.get(op_str.split()[0])
        if handler is None:
            raise ValueError(f"Unknown or unimplemented operation: {op_str!r}")

        try:
            result = handler(op_str, val_list, lib_ref_list, libs, args)
        except Exception as e:
            lhs = ",".join(out_names)
            raise type(e)(f"{e} [in `{lhs} = {op_str}; {', '.join(raw_args)}`]") from e

        if len(out_names) == 1:
            if isinstance(result, (list, tuple)):
                if len(result) != 1:
                    raise ValueError(
                        f"Op {op_str!r} returned {len(result)} values "
                        f"but only one output was expected"
                    )
                result = result[0]
            vars_dict[out_names[0]] = result
        else:
            n_ret = len(result) if isinstance(result, (list, tuple)) else 1
            if n_ret != len(out_names):
                raise ValueError(
                    f"Op {op_str!r} returned {n_ret} value(s) "
                    f"but {len(out_names)} were expected"
                )
            for name, val in zip(out_names, result):
                vars_dict[name] = val

        if intermediates is not None:
            for name in out_names:
                intermediates[int(name[1:])] = vars_dict[name]

    raise ValueError("Body has no return statement")  # unreachable; checked at parse


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    code: str,
    *args: jnp.ndarray,
    intermediates: Optional[Iterable[int]] = None,
) -> Union[jnp.ndarray, List[jnp.ndarray], Tuple[Any, Dict[int, jnp.ndarray]]]:
    """
    Evaluate SSA-generated XLA IR code with the given input arrays.

    Args:
        code: The IR string generated by `Expr.code`.
        *args: Input arrays corresponding to $0, $1, ...
        intermediates: optional iterable of variable ids (the `N` in `%N`)
            to capture. When given, returns `(result, captured)` where
            `captured` maps each requested id to its array.

    Returns:
        The output array, a list of arrays for multi-output expressions,
        or `(output, captured)` when `intermediates` is given.
    """
    body, libs = parse_ir(code)
    captured: Optional[Dict[int, jnp.ndarray]] = {} if intermediates is not None else None
    results = eval_body(body, libs, list(args), intermediates=captured)
    result: Any = results[0] if len(results) == 1 else results
    if intermediates is None:
        return result
    missing = [i for i in intermediates if i not in captured]
    if missing:
        raise ValueError(f"Intermediate variable(s) {missing} were never assigned")
    return result, {i: captured[i] for i in intermediates}


def evaluate_multi(code: str, *args: jnp.ndarray) -> List[jnp.ndarray]:
    """Evaluate and return all outputs as a list."""
    body, libs = parse_ir(code)
    return eval_body(body, libs, list(args))
