"""
Operation handlers for the SSA XLA IR evaluator.

Each handler is registered via @register_op(name) from eval.py.
Importing this module automatically populates the operation registry.

Handlers receive:
    op_str      the full op string (e.g. "transpose [1, 0]")
    vals        resolved non-library arguments, in order
    lib_refs    indices of `@i` library arguments, in order
    libs        all parsed library bodies
    parent_args the `$i` arguments of the enclosing body
"""

import ast
import re
import jax
import jax.numpy as jnp
import numpy as np

from eval import register_op, eval_body, _lib_output_count


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "int": jnp.int32,
    "float": jnp.float32,
}


def _parse_shape(shape_str: str) -> tuple:
    return tuple(int(x) for x in shape_str.strip("[]").split(",") if x.strip())


def _parse_bool(s: str) -> bool:
    s = s.strip().lower()
    if s not in ("true", "false"):
        raise ValueError(f"Expected 'true' or 'false', got {s!r}")
    return s == "true"


def _opt_int(parts, i, default=None):
    """Optional integer parameter: new-style IR prints it, legacy IR omits it."""
    return int(parts[i]) if len(parts) > i else default


# ---------------------------------------------------------------------------
# Operation helpers
# ---------------------------------------------------------------------------

def _eval_const(dtype_str: str, shape_str: str, val_str: str):
    dtype = _DTYPE_MAP.get(dtype_str, jnp.float32)
    return jnp.full(_parse_shape(shape_str), int(val_str), dtype=dtype)


def _eval_zeros(dtype_str: str, shape_str: str):
    dtype = _DTYPE_MAP.get(dtype_str, jnp.float32)
    return jnp.zeros(_parse_shape(shape_str), dtype=dtype)


def _eval_transpose(perm_str: str, x):
    return jnp.transpose(x, axes=_parse_shape(perm_str))


def _eval_dot_general(contract_len: int, batch_len: int, x, y):
    C, B = contract_len, batch_len
    dimension_numbers = (
        (tuple(range(C)), tuple(range(C))),                    # contracting dims
        (tuple(range(C, C + B)), tuple(range(C, C + B))),      # batch dims
    )
    return jax.lax.dot_general(x, y, dimension_numbers=dimension_numbers)


def _eval_broadcast_impl(bools_str: str, x):
    bools = [_parse_bool(p) for p in bools_str.strip("[]").split(",") if p.strip()]
    n_input_dims = sum(bools)
    if n_input_dims != x.ndim:
        raise ValueError(
            f"Broadcast: input has {x.ndim} dims but bools {bools} expect {n_input_dims}"
        )
    new_shape, input_dim = [], 0
    for keep in bools:
        if keep:
            new_shape.append(x.shape[input_dim])
            input_dim += 1
        else:
            new_shape.append(1)
    return x.reshape(new_shape)


def _eval_scatter(x, y, indices):
    """
    Scatter with Lean semantics (`DirectImpl.scatter`): the *first* matching
    update wins. Vectorized in two steps: for each target slot, find the
    smallest update index writing to it (an order-independent min-reduction),
    then select between that update and the original value.
    """
    n = y.shape[0]
    if n == 0:
        return x
    idx = tuple(jnp.asarray(i) for i in indices)
    winner = jnp.full(x.shape, n, dtype=jnp.int32).at[idx].min(
        jnp.arange(n, dtype=jnp.int32)
    )
    update = y[jnp.minimum(winner, n - 1)]
    return jnp.where(winner < n, update, x)


def _eval_scatter_add(x, y, indices):
    """
    Scatter-add with Lean semantics (`DirectImpl.scatter_add`): every update
    whose coordinates match a slot is summed into the base. Indices wrap modulo
    each axis (like `Fin.intCast`), are flattened with row-major strides, and
    accumulated with `at[].add`, which is order-independent and jit/grad-friendly.
    """
    n = y.shape[0]
    if n == 0:
        return x
    x = jnp.asarray(x)
    if 0 in x.shape:
        return x
    lin = jnp.zeros(n, dtype=jnp.int32)
    stride = 1
    for axis in range(x.ndim - 1, -1, -1):
        i = jnp.asarray(indices[axis], dtype=jnp.int32) % x.shape[axis]
        lin = lin + i * stride
        stride *= x.shape[axis]
    acc = jnp.zeros((stride,), dtype=x.dtype).at[lin].add(y)
    return x + acc.reshape(x.shape)


def _eval_gather(x, indices):
    return x[tuple(jnp.asarray(i) for i in indices)]


def _apply_cum(op, x, axis: int, reverse: bool):
    """Cumulative op with optional reversal (scan from the right)."""
    if reverse:
        x = jnp.flip(x, axis=axis)
    out = op(x, axis=axis)
    return jnp.flip(out, axis=axis) if reverse else out


# ---------------------------------------------------------------------------
# Registered operations
# ---------------------------------------------------------------------------

# -- Constants & constructors ------------------------------------------------

@register_op("const")
def _eval_const_op(op_str, vals, lib_refs, libs, parent_args):
    m = re.fullmatch(r"const\s+(\S+)\s+(\[.*?\])\s+(\S+)", op_str)
    if not m:
        raise ValueError(f"Invalid const op: {op_str!r}")
    return _eval_const(*m.groups())


@register_op("zeros")
def _eval_zeros_op(op_str, vals, lib_refs, libs, parent_args):
    m = re.fullmatch(r"zeros\s+(\S+)\s+(\[.*?\])", op_str)
    if not m:
        raise ValueError(f"Invalid zeros op: {op_str!r}")
    return _eval_zeros(*m.groups())


@register_op("iota")
def _eval_iota_op(op_str, vals, lib_refs, libs, parent_args):
    return jnp.arange(int(op_str.split()[1]), dtype=jnp.int32)


@register_op("empty")
def _eval_empty(op_str, vals, lib_refs, libs, parent_args):
    # The IR does not encode shape/dtype for `empty`.
    raise ValueError("empty: cannot determine shape/dtype from IR alone")


# -- Element-wise unary ------------------------------------------------------

@register_op("neg")
def _eval_neg(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return -x

@register_op("sqrt")
def _eval_sqrt(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jnp.sqrt(x)

@register_op("abs")
def _eval_abs(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jnp.abs(x)

@register_op("cos")
def _eval_cos(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jnp.cos(x)

@register_op("sin")
def _eval_sin(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jnp.sin(x)

@register_op("exp")
def _eval_exp(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jnp.exp(x)

@register_op("log")
def _eval_log(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jnp.log(x)

@register_op("tanh")
def _eval_tanh(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jnp.tanh(x)

@register_op("ceil")
def _eval_ceil(op_str, vals, lib_refs, libs, parent_args):
    # Note: the Lean op signature declares an int output; kept as float here
    # to preserve differentiability (XLA `ceil` semantics).
    x, = vals; return jnp.ceil(x)

@register_op("floor")
def _eval_floor(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jnp.floor(x)

@register_op("acos")
def _eval_acos(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jnp.arccos(x)

@register_op("acosh")
def _eval_acosh(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jnp.arccosh(x)

@register_op("asin")
def _eval_asin(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jnp.arcsin(x)

@register_op("asinh")
def _eval_asinh(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jnp.arcsinh(x)

@register_op("atan")
def _eval_atan(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jnp.arctan(x)

@register_op("atanh")
def _eval_atanh(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jnp.arctanh(x)

@register_op("cbrt")
def _eval_cbrt(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jnp.cbrt(x)

@register_op("cosh")
def _eval_cosh(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jnp.cosh(x)

@register_op("erf")
def _eval_erf(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jax.scipy.special.erf(x)

@register_op("erf_inv")
def _eval_erf_inv(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jax.scipy.special.erfinv(x)

@register_op("erfc")
def _eval_erfc(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jax.scipy.special.erfc(x)

@register_op("exp2")
def _eval_exp2(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jnp.exp2(x)

@register_op("expm1")
def _eval_expm1(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jnp.expm1(x)

@register_op("bessel_i0e")
def _eval_bessel_i0e(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jax.scipy.special.i0e(x)

@register_op("bessel_i1e")
def _eval_bessel_i1e(op_str, vals, lib_refs, libs, parent_args):
    x, = vals; return jax.scipy.special.i1e(x)


# -- Element-wise binary -----------------------------------------------------

@register_op("add")
def _eval_add(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals; return x + y

@register_op("sub")
def _eval_sub(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals; return x - y

@register_op("mul")
def _eval_mul(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals; return x * y

@register_op("div")
def _eval_div(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals; return x / y

@register_op("eq")
def _eval_eq(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals; return (x == y).astype(jnp.int32)

@register_op("lt")
def _eval_lt(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals; return (x < y).astype(jnp.int32)

@register_op("gt")
def _eval_gt(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals; return (x > y).astype(jnp.int32)

@register_op("max")
def _eval_max(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals; return jnp.maximum(x, y)

@register_op("min")
def _eval_min(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals; return jnp.minimum(x, y)

@register_op("mod")
def _eval_mod(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals; return jnp.mod(x, y)

@register_op("div_int")
def _eval_div_int(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals; return jnp.floor_divide(x, y)

@register_op("and")
def _eval_and(op_str, vals, lib_refs, libs, parent_args):
    x, y = vals; return jnp.bitwise_and(x, y)


# -- Reductions & cumulative -------------------------------------------------

@register_op("sum")
def _eval_sum(op_str, vals, lib_refs, libs, parent_args):
    n = int(op_str.split()[1])
    x, = vals
    return jnp.sum(x, axis=tuple(range(n)))


@register_op("cumsum")
def _eval_cumsum(op_str, vals, lib_refs, libs, parent_args):
    # `Tensor.cumsum` recurses to the innermost axis.
    x, = vals
    if x.ndim == 0:
        return x
    return jnp.cumsum(x, axis=-1)


@register_op("argmax")
def _eval_argmax(op_str, vals, lib_refs, libs, parent_args):
    axis = int(op_str.split()[1])
    x, = vals
    return jnp.argmax(x, axis=axis).astype(jnp.int32)


@register_op("argmin")
def _eval_argmin(op_str, vals, lib_refs, libs, parent_args):
    axis = int(op_str.split()[1])
    x, = vals
    return jnp.argmin(x, axis=axis).astype(jnp.int32)


def _cum_parts(op_str, default_axis=-1):
    parts = op_str.split()
    axis = int(parts[1]) if len(parts) > 1 else default_axis
    reverse = _parse_bool(parts[2]) if len(parts) > 2 else False
    return axis, reverse


@register_op("cummax")
def _eval_cummax(op_str, vals, lib_refs, libs, parent_args):
    axis, reverse = _cum_parts(op_str)
    x, = vals
    return _apply_cum(jnp.maximum.accumulate, x, axis, reverse)


@register_op("cummin")
def _eval_cummin(op_str, vals, lib_refs, libs, parent_args):
    axis, reverse = _cum_parts(op_str)
    x, = vals
    return _apply_cum(jnp.minimum.accumulate, x, axis, reverse)


@register_op("cumprod")
def _eval_cumprod(op_str, vals, lib_refs, libs, parent_args):
    axis, reverse = _cum_parts(op_str)
    x, = vals
    return _apply_cum(jnp.cumprod, x, axis, reverse)


@register_op("cumlogsumexp")
def _eval_cumlogsumexp(op_str, vals, lib_refs, libs, parent_args):
    axis, reverse = _cum_parts(op_str)
    x, = vals
    def lse(a, axis):
        c = jnp.cumsum(jnp.exp(a), axis=axis)
        return jnp.log(c)
    return _apply_cum(lse, x, axis, reverse)


# -- Shape manipulation ------------------------------------------------------

@register_op("transpose")
def _eval_transpose_op(op_str, vals, lib_refs, libs, parent_args):
    x, = vals
    return _eval_transpose(op_str[len("transpose "):], x)


@register_op("broadcast")
def _eval_broadcast(op_str, vals, lib_refs, libs, parent_args):
    x, = vals
    return _eval_broadcast_impl(op_str[len("broadcast "):], x)


@register_op("concat")
def _eval_concat(op_str, vals, lib_refs, libs, parent_args):
    parts = op_str.split()
    axis = _opt_int(parts, 1, default=0)  # legacy IR omits the axis
    return jnp.concatenate(vals, axis=axis)


@register_op("flatten")
def _eval_flatten(op_str, vals, lib_refs, libs, parent_args):
    x, = vals
    return x.reshape(-1)


@register_op("unflatten")
def _eval_unflatten(op_str, vals, lib_refs, libs, parent_args):
    x, = vals
    return x.reshape(_parse_shape(op_str[len("unflatten "):]))


@register_op("id")
def _eval_id(op_str, vals, lib_refs, libs, parent_args):
    x, = vals
    return x


@register_op("convert_type")
def _eval_convert_type(op_str, vals, lib_refs, libs, parent_args):
    x, = vals
    parts = op_str.split()
    if len(parts) >= 3:      # new-style: "convert_type <src> <dst>"
        return x.astype(_DTYPE_MAP[parts[2]])
    if len(parts) == 2:      # "convert_type <dst>"
        return x.astype(_DTYPE_MAP[parts[1]])
    # Legacy IR does not encode the target dtype; fall back to the
    # direction implied by the input dtype.
    if jnp.issubdtype(x.dtype, jnp.integer):
        return x.astype(jnp.float32)
    return x.astype(jnp.int32)


# -- Linear algebra ----------------------------------------------------------

@register_op("cholesky")
def _eval_cholesky(op_str, vals, lib_refs, libs, parent_args):
    x, = vals
    return jnp.linalg.cholesky(x)


@register_op("det")
def _eval_det(op_str, vals, lib_refs, libs, parent_args):
    x, = vals
    return jnp.linalg.det(x)


@register_op("eigvals")
def _eval_eigvals(op_str, vals, lib_refs, libs, parent_args):
    x, = vals
    return jnp.linalg.eigvals(x).astype(jnp.float32)


@register_op("eigvalsh")
def _eval_eigvalsh(op_str, vals, lib_refs, libs, parent_args):
    x, = vals
    return jnp.linalg.eigvalsh(x).astype(jnp.float32)


@register_op("eigvecs")
def _eval_eigvecs(op_str, vals, lib_refs, libs, parent_args):
    x, = vals
    _, v = jnp.linalg.eig(x)
    return v.astype(jnp.float32)


@register_op("eigvecsh")
def _eval_eigvecsh(op_str, vals, lib_refs, libs, parent_args):
    x, = vals
    _, v = jnp.linalg.eigh(x)
    return v.astype(jnp.float32)


@register_op("dot_general")
def _eval_dot_general_op(op_str, vals, lib_refs, libs, parent_args):
    parts = op_str.split()
    x, y = vals
    return _eval_dot_general(int(parts[1]), int(parts[2]), x, y)


@register_op("einsum")
def _eval_einsum(op_str, vals, lib_refs, libs, parent_args):
    """Format: `einsum [[i,j,...], [k,l,...], ...] n; arg1, arg2, ...`

    Contracts the first `n` axes of the ambient shape; the remaining axes
    (in order) form the output. Uses opt_einsum-style operand notation so
    axis labels never run out of letters.
    """
    m = re.fullmatch(r"einsum\s+(\[.*\])\s+(\d+)", op_str)
    if not m:
        raise ValueError(f"Invalid einsum op: {op_str!r}")
    specs = ast.literal_eval(m.group(1))
    nsum = int(m.group(2))

    max_idx = max((i for spec in specs for i in spec), default=-1)
    operands = []
    for arg, spec in zip(vals, specs):
        operands += [arg, list(spec)]
    operands.append(list(range(nsum, max_idx + 1)))
    return jnp.einsum(*operands)


# -- Selection & indexing ----------------------------------------------------

@register_op("where")
def _eval_where(op_str, vals, lib_refs, libs, parent_args):
    c, x, y = vals
    return jnp.where(c != 0, x, y)


@register_op("scatter")
def _eval_scatter_op(op_str, vals, lib_refs, libs, parent_args):
    return _eval_scatter(vals[0], vals[1], vals[2:])


@register_op("scatter_add")
def _eval_scatter_add_op(op_str, vals, lib_refs, libs, parent_args):
    return _eval_scatter_add(vals[0], vals[1], vals[2:])


@register_op("gather")
def _eval_gather_op(op_str, vals, lib_refs, libs, parent_args):
    return _eval_gather(vals[0], vals[1:])


@register_op("sorted")
def _eval_sorted(op_str, vals, lib_refs, libs, parent_args):
    x, = vals
    return jnp.sort(x, axis=-1)


# -- Control flow ------------------------------------------------------------

@register_op("repeat")
def _eval_repeat(op_str, vals, lib_refs, libs, parent_args):
    """`repeat; @f, n, carry..., aux...` — iterate `f` |n| times on the carry.

    The carry size is read off the library's return arity. Uses
    `jax.lax.fori_loop`, so the trip count may be a traced value.
    """
    lib_body = libs[lib_refs[0]]
    count = jnp.abs(jnp.asarray(vals[0]))
    carry_size = _lib_output_count(lib_body)
    init_carry = vals[1:1 + carry_size]
    aux = vals[1 + carry_size:]

    def body_fn(i, carry):
        carry_args = [carry] if carry_size == 1 else list(carry)
        result = eval_body(lib_body, libs, carry_args + list(aux))
        return result[0] if carry_size == 1 else tuple(result)

    init = init_carry[0] if carry_size == 1 else tuple(init_carry)
    result = jax.lax.fori_loop(0, count, body_fn, init)
    return result if carry_size == 1 else list(result)


@register_op("vmap")
def _eval_vmap(op_str, vals, lib_refs, libs, parent_args):
    """`vmap k; @f, batched..., aux...` — map `f` over the leading axis of
    the first `k` inputs; the rest are passed through unchanged.

    Legacy IR omits `k`; then every input is treated as batched.
    """
    parts = op_str.split()
    n_batched = _opt_int(parts, 1, default=len(vals))
    batched, aux = vals[:n_batched], vals[n_batched:]
    lib_body = libs[lib_refs[0]]

    if not batched:
        result = eval_body(lib_body, libs, list(aux))
        return result[0] if len(result) == 1 else result

    def fn(*xs):
        return eval_body(lib_body, libs, list(xs) + list(aux))

    result = jax.vmap(fn)(*batched)
    return result[0] if len(result) == 1 else list(result)


@register_op("call")
def _eval_call(op_str, vals, lib_refs, libs, parent_args):
    result = eval_body(libs[lib_refs[0]], libs, vals)
    return result[0] if len(result) == 1 else result
