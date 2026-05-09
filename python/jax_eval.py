'''
JAX-based evaluator for SSA-generated XLA IR.

This evaluator parses the IR strings produced by `Expr.code` in SSA/Core.lean
and evaluates them using JAX operations, matching the Lean `DirectImpl` semantics.

Example::

    from python.jax_eval import evaluate
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

    from python.jax_eval import register_op
    import jax.numpy as jnp

    @register_op("my_op")
    def _eval_my_op(op_str, vals, lib_refs, libs, parent_args):
        x, y = vals()
        return jnp.do_something(x, y)
'''

import re
from typing import List, Tuple, Dict, Callable, Any
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Operation registry
# ---------------------------------------------------------------------------

_OP_REGISTRY: Dict[str, Callable] = {}
_OP_PREFIX_REGISTRY: List[Tuple[str, Callable]] = []


def register_op(name: str):
    """Decorator to register an operation handler by exact name."""
    def decorator(fn: Callable):
        _OP_REGISTRY[name] = fn
        return fn
    return decorator


def register_op_prefix(prefix: str):
    """Decorator to register an operation handler by name prefix."""
    def decorator(fn: Callable):
        _OP_PREFIX_REGISTRY.append((prefix, fn))
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
# Operation helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "int": jnp.int32,
    "float": jnp.float32,
}


def _eval_const(dtype_str: str, shape_str: str, val_str: str):
    dtype = _DTYPE_MAP.get(dtype_str, jnp.float32)
    shape = tuple(int(x) for x in shape_str.strip("[]").split(",") if x.strip())
    val = int(val_str)
    return jnp.full(shape, val, dtype=dtype)


def _eval_zeros(dtype_str: str, shape_str: str):
    dtype = _DTYPE_MAP.get(dtype_str, jnp.float32)
    shape = tuple(int(x) for x in shape_str.strip("[]").split(",") if x.strip())
    return jnp.zeros(shape, dtype=dtype)


def _eval_iota(n: int):
    return jnp.arange(n, dtype=jnp.int32)


def _eval_transpose(perm_str: str, x):
    perm = tuple(int(x.strip()) for x in perm_str.strip("[]").split(",") if x.strip())
    return jnp.transpose(x, axes=perm)


def _eval_dot_general(contract_len: int, batch_len: int, x, y):
    C = contract_len
    B = batch_len
    lhs_contract = tuple(range(C))
    rhs_contract = tuple(range(C))
    lhs_batch = tuple(range(C, C + B))
    rhs_batch = tuple(range(C, C + B))
    dimension_numbers = ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))
    return jax.lax.dot_general(x, y, dimension_numbers=dimension_numbers)


def _eval_broadcast_impl(bools_str: str, x):
    bools = [p.strip().lower() == "true" for p in bools_str.strip("[]").split(",") if p.strip()]
    n_input_dims = sum(bools)
    if n_input_dims != x.ndim:
        raise ValueError(
            f"Broadcast: input has {x.ndim} dims but bools {bools} expect {n_input_dims}"
        )
    new_shape = []
    input_dim = 0
    for keep in bools:
        if keep:
            new_shape.append(x.shape[input_dim])
            input_dim += 1
        else:
            new_shape.append(1)
    return x.reshape(new_shape)


def _eval_scatter(x, y, indices):
    """
    Scatter with Lean semantics: first matching update wins.
    We iterate in reverse so that earlier updates overwrite later ones.
    """
    out = jnp.array(x)
    n = y.shape[0]
    ndim = len(indices)
    for k in reversed(range(n)):
        idx = tuple(int(indices[d][k]) for d in range(ndim))
        out = out.at[idx].set(y[k])
    return out


def _eval_gather(x, indices):
    idx = tuple(indices)
    return x[idx]


# ---------------------------------------------------------------------------
# Registered operations
# ---------------------------------------------------------------------------

# -- Constants & constructors ------------------------------------------------

@register_op_prefix("const ")
def _eval_const_op(op_str, vals, lib_refs, libs, parent_args):
    m = re.match(r"const\s+(\S+)\s+(\[.*?\])\s+(\S+)", op_str)
    if not m:
        raise ValueError(f"Invalid const op: {op_str!r}")
    dtype, shape_str, val_str = m.groups()
    return _eval_const(dtype, shape_str, val_str)


@register_op_prefix("zeros ")
def _eval_zeros_op(op_str, vals, lib_refs, libs, parent_args):
    m = re.match(r"zeros\s+(\S+)\s+(\[.*?\])", op_str)
    if not m:
        raise ValueError(f"Invalid zeros op: {op_str!r}")
    dtype, shape_str = m.groups()
    return _eval_zeros(dtype, shape_str)


@register_op_prefix("iota ")
def _eval_iota_op(op_str, vals, lib_refs, libs, parent_args):
    n = int(op_str.split()[1])
    return _eval_iota(n)


# -- Element-wise unary ------------------------------------------------------

@register_op("neg")
def _eval_neg(op_str, vals, lib_refs, libs, parent_args):
    x, = vals(); return -x

@register_op("sqrt")
def _eval_sqrt(op_str, vals, lib_refs, libs, parent_args):
    x, = vals(); return jnp.sqrt(x)

@register_op("abs")
def _eval_abs(op_str, vals, lib_refs, libs, parent_args):
    x, = vals(); return jnp.abs(x)

@register_op("cos")
def _eval_cos(op_str, vals, lib_refs, libs, parent_args):
    x, = vals(); return jnp.cos(x)

@register_op("sin")
def _eval_sin(op_str, vals, lib_refs, libs, parent_args):
    x, = vals(); return jnp.sin(x)

@register_op("exp")
def _eval_exp(op_str, vals, lib_refs, libs, parent_args):
    x, = vals(); return jnp.exp(x)

@register_op("log")
def _eval_log(op_str, vals, lib_refs, libs, parent_args):
    x, = vals(); return jnp.log(x)

@register_op("tanh")
def _eval_tanh(op_str, vals, lib_refs, libs, parent_args):
    x, = vals(); return jnp.tanh(x)

@register_op("ceil")
def _eval_ceil(op_str, vals, lib_refs, libs, parent_args):
    x, = vals(); return jnp.ceil(x)

@register_op("floor")
def _eval_floor(op_str, vals, lib_refs, libs, parent_args):
    x, = vals(); return jnp.floor(x)


# -- Element-wise binary -----------------------------------------------------

@register_op("add")
def _eval_add(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals(); return x + y

@register_op("sub")
def _eval_sub(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals(); return x - y

@register_op("mul")
def _eval_mul(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals(); return x * y

@register_op("div")
def _eval_div(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals(); return x / y

@register_op("eq")
def _eval_eq(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals(); return (x == y).astype(jnp.int32)

@register_op("lt")
def _eval_lt(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals(); return (x < y).astype(jnp.int32)

@register_op("gt")
def _eval_gt(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals(); return (x > y).astype(jnp.int32)

@register_op("max")
def _eval_max(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals(); return jnp.maximum(x, y)

@register_op("min")
def _eval_min(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals(); return jnp.minimum(x, y)


# -- Reductions & cumulative -------------------------------------------------

@register_op_prefix("sum ")
def _eval_sum(op_str, vals, lib_refs, libs, parent_args):
    n = int(op_str.split()[1])
    x, = vals()
    return jnp.sum(x, axis=tuple(range(n)))


@register_op("cumsum")
def _eval_cumsum(op_str, vals, lib_refs, libs, parent_args):
    x, = vals()
    if x.ndim == 0:
        return x
    return jnp.cumsum(x, axis=-1)


# -- Shape manipulation ------------------------------------------------------

@register_op_prefix("transpose ")
def _eval_transpose_op(op_str, vals, lib_refs, libs, parent_args):
    perm_str = op_str[len("transpose "):]
    x, = vals()
    return _eval_transpose(perm_str, x)


@register_op_prefix("braodcast ")
def _eval_broadcast_typo(op_str, vals, lib_refs, libs, parent_args):
    bools_str = op_str[len("braodcast "):]
    x, = vals()
    return _eval_broadcast_impl(bools_str, x)


@register_op_prefix("broadcast ")
def _eval_broadcast(op_str, vals, lib_refs, libs, parent_args):
    bools_str = op_str[len("broadcast "):]
    x, = vals()
    return _eval_broadcast_impl(bools_str, x)


@register_op("concat")
def _eval_concat(op_str, vals, lib_refs, libs, parent_args):
    xs = vals()
    return jnp.concatenate(xs, axis=0)


# -- Linear algebra ----------------------------------------------------------

@register_op_prefix("dot_general ")
def _eval_dot_general_op(op_str, vals, lib_refs, libs, parent_args):
    parts = op_str.split()
    contract_len = int(parts[1])
    batch_len = int(parts[2])
    x, y = vals()
    return _eval_dot_general(contract_len, batch_len, x, y)


# -- Selection & indexing ----------------------------------------------------

@register_op("where")
def _eval_where(op_str, vals, lib_refs, libs, parent_args):
    c, x, y = vals()
    return jnp.where(c != 0, x, y)


@register_op("scatter")
def _eval_scatter_op(op_str, vals, lib_refs, libs, parent_args):
    arr = vals()
    return _eval_scatter(arr[0], arr[1], arr[2:])


@register_op("gather")
def _eval_gather_op(op_str, vals, lib_refs, libs, parent_args):
    arr = vals()
    return _eval_gather(arr[0], arr[1:])


@register_op("sorted")
def _eval_sorted(op_str, vals, lib_refs, libs, parent_args):
    x, = vals()
    return jnp.sort(x, axis=-1)


# -- Control flow ------------------------------------------------------------

@register_op("repeat")
def _eval_repeat(op_str, vals, lib_refs, libs, parent_args):
    lib_idx = lib_refs()[0]
    count = vals()[0]
    inputs = vals()[1:]
    lib_body = libs[lib_idx]
    carry_size = _lib_output_count(lib_body)
    init_carry = inputs[:carry_size]
    aux = inputs[carry_size:]

    def body_fn(i, carry):
        carry_args = [carry] if carry_size == 1 else list(carry)
        all_args = carry_args + list(aux)
        result = eval_body(lib_body, libs, all_args)
        if len(result) == 1:
            return result[0]
        return tuple(result)

    count_int = int(jnp.asarray(count).item())
    init = init_carry[0] if carry_size == 1 else tuple(init_carry)
    result = jax.lax.fori_loop(0, count_int, body_fn, init)
    if carry_size == 1:
        return result
    return list(result)


@register_op_prefix("call ")
def _eval_call(op_str, vals, lib_refs, libs, parent_args):
    lib_idx = int(op_str.split()[1][1:])
    call_args = vals()
    lib_body = libs[lib_idx]
    result = eval_body(lib_body, libs, call_args)
    if len(result) == 1:
        return result[0]
    return result


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

        def vals():
            return [r for r in resolved if not (isinstance(r, tuple) and r[0] == "lib")]

        def lib_refs():
            return [r[1] for r in resolved if isinstance(r, tuple) and r[0] == "lib"]

        # Dispatch via registry
        handler = _OP_REGISTRY.get(op_str)
        if handler is None:
            for prefix, prefix_handler in _OP_PREFIX_REGISTRY:
                if op_str.startswith(prefix):
                    handler = prefix_handler
                    break

        if handler is None:
            raise ValueError(f"Unknown or unimplemented operation: {op_str!r}")

        result = handler(op_str, vals, lib_refs, libs, args)

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
