"""
Operation handlers for the SSA XLA IR evaluator.

Each handler is registered via @register_op(name) from eval.py.
Importing this module automatically populates the operation registry.
"""

import re
import jax
import jax.numpy as jnp
import numpy as np

from eval import register_op, eval_body, _lib_output_count


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

@register_op("const")
def _eval_const_op(op_str, vals, lib_refs, libs, parent_args):
    m = re.match(r"const\s+(\S+)\s+(\[.*?\])\s+(\S+)", op_str)
    if not m:
        raise ValueError(f"Invalid const op: {op_str!r}")
    dtype, shape_str, val_str = m.groups()
    return _eval_const(dtype, shape_str, val_str)


@register_op("zeros")
def _eval_zeros_op(op_str, vals, lib_refs, libs, parent_args):
    m = re.match(r"zeros\s+(\S+)\s+(\[.*?\])", op_str)
    if not m:
        raise ValueError(f"Invalid zeros op: {op_str!r}")
    dtype, shape_str = m.groups()
    return _eval_zeros(dtype, shape_str)


@register_op("iota")
def _eval_iota_op(op_str, vals, lib_refs, libs, parent_args):
    n = int(op_str.split()[1])
    return _eval_iota(n)


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

@register_op("cummax")
def _eval_cummax(op_str, vals, lib_refs, libs, parent_args):
    parts = op_str.split()
    axis = int(parts[1])
    x, = vals
    return jnp.maximum.accumulate(x, axis=axis)

@register_op("cummin")
def _eval_cummin(op_str, vals, lib_refs, libs, parent_args):
    parts = op_str.split()
    axis = int(parts[1])
    x, = vals
    return jnp.minimum.accumulate(x, axis=axis)

@register_op("cumprod")
def _eval_cumprod(op_str, vals, lib_refs, libs, parent_args):
    parts = op_str.split()
    axis = int(parts[1])
    x, = vals
    return jnp.cumprod(x, axis=axis)

@register_op("cumlogsumexp")
def _eval_cumlogsumexp(op_str, vals, lib_refs, libs, parent_args):
    parts = op_str.split()
    axis = int(parts[1])
    x, = vals
    return jnp.log(jnp.cumsum(jnp.exp(x), axis=axis))


# -- Shape manipulation ------------------------------------------------------

@register_op("transpose")
def _eval_transpose_op(op_str, vals, lib_refs, libs, parent_args):
    perm_str = op_str[len("transpose "):]
    x, = vals
    return _eval_transpose(perm_str, x)


@register_op("broadcast")
def _eval_broadcast_typo(op_str, vals, lib_refs, libs, parent_args):
    bools_str = op_str[len("broadcast "):]
    x, = vals
    return _eval_broadcast_impl(bools_str, x)


@register_op("broadcast")
def _eval_broadcast(op_str, vals, lib_refs, libs, parent_args):
    bools_str = op_str[len("broadcast "):]
    x, = vals
    return _eval_broadcast_impl(bools_str, x)


@register_op("concat")
def _eval_concat(op_str, vals, lib_refs, libs, parent_args):
    xs = vals
    return jnp.concatenate(xs, axis=0)


# -- Linear algebra ----------------------------------------------------------

@register_op("cholesky")
def _eval_cholesky(op_str, vals, lib_refs, libs, parent_args):
    x, = vals
    return jnp.linalg.cholesky(x)

@register_op("eigvals")
def _eval_eigvals(op_str, vals, lib_refs, libs, parent_args):
    x, = vals
    w = jnp.linalg.eigvals(x)
    return w.astype(jnp.float32)

@register_op("eigvalsh")
def _eval_eigvalsh(op_str, vals, lib_refs, libs, parent_args):
    x, = vals
    return jnp.linalg.eigvalsh(x).astype(jnp.float32)

@register_op("eigvecs")
def _eval_eigvecs(op_str, vals, lib_refs, libs, parent_args):
    x, = vals
    w, v = jnp.linalg.eig(x)
    return v.astype(jnp.float32)

@register_op("eigvecsh")
def _eval_eigvecsh(op_str, vals, lib_refs, libs, parent_args):
    x, = vals
    w, v = jnp.linalg.eigh(x)
    return v.astype(jnp.float32)

@register_op("convert_type")
def _eval_convert_type(op_str, vals, lib_refs, libs, parent_args):
    x, = vals
    # IR doesn't encode target dtype; default to float32 for int->float
    if jnp.issubdtype(x.dtype, jnp.integer):
        return x.astype(jnp.float32)
    return x.astype(jnp.int32)

@register_op("empty")
def _eval_empty(op_str, vals, lib_refs, libs, parent_args):
    # IR doesn't encode shape/dtype; this is a placeholder
    raise ValueError("empty: cannot determine shape/dtype from IR alone")

@register_op("dot_general")
def _eval_dot_general_op(op_str, vals, lib_refs, libs, parent_args):
    parts = op_str.split()
    contract_len = int(parts[1])
    batch_len = int(parts[2])
    x, y = vals
    return _eval_dot_general(contract_len, batch_len, x, y)


# -- Selection & indexing ----------------------------------------------------

@register_op("where")
def _eval_where(op_str, vals, lib_refs, libs, parent_args):
    c, x, y = vals
    return jnp.where(c != 0, x, y)


@register_op("scatter")
def _eval_scatter_op(op_str, vals, lib_refs, libs, parent_args):
    arr = vals
    return _eval_scatter(arr[0], arr[1], arr[2:])


@register_op("gather")
def _eval_gather_op(op_str, vals, lib_refs, libs, parent_args):
    arr = vals
    return _eval_gather(arr[0], arr[1:])


@register_op("sorted")
def _eval_sorted(op_str, vals, lib_refs, libs, parent_args):
    x, = vals
    return jnp.sort(x, axis=-1)


# -- Control flow ------------------------------------------------------------

@register_op("repeat")
def _eval_repeat(op_str, vals, lib_refs, libs, parent_args):
    lib_idx = lib_refs[0]
    count = vals[0]
    inputs = vals[1:]
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


@register_op("vmap")
def _eval_vmap(op_str, vals, lib_refs, libs, parent_args):
    """Vectorize a library function over the leading axis of inputs."""
    lib_idx = lib_refs[0]
    inputs = vals
    lib_body = libs[lib_idx]
    
    if not inputs:
        return eval_body(lib_body, libs, [])
    
    batch_size = int(inputs[0].shape[0])
    results = []
    
    for i in range(batch_size):
        slice_args = [inp[i] for inp in inputs]
        result = eval_body(lib_body, libs, slice_args)
        if len(result) == 1:
            results.append(result[0])
        else:
            results.append(tuple(result))
    
    if not results:
        return results
    if isinstance(results[0], tuple):
        return [jnp.stack([r[j] for r in results], axis=0) for j in range(len(results[0]))]
    return jnp.stack(results, axis=0)


@register_op("einsum")
def _eval_einsum(op_str, vals, lib_refs, libs, parent_args):
    """Parse einsum spec and evaluate using JAX."""
    # Format: einsum [[i,j,...], [k,l,...], ...] n; arg1, arg2, ...
    m = re.match(r"einsum\s+(\[\[.*\]\])\s+(\d+)", op_str)
    if not m:
        raise ValueError(f"Invalid einsum op: {op_str!r}")
    
    specs_str = m.group(1)
    nsum = int(m.group(2))
    
    # Parse specs: [[2,1], [1,0], ...]
    specs = []
    # specs_str is like "[[1, 0], [0]]"; strip outer brackets and split
    inner = specs_str[1:-1].strip()
    parts = [p.strip("[] ") for p in inner.split("], [")]
    for p in parts:
        idxs = [int(x.strip()) for x in p.split(",") if x.strip()]
        if idxs:
            specs.append(idxs)
    
    args = vals
    
    # Find max axis index
    max_idx = max((max(s) for s in specs if s), default=-1)
    
    # Map indices to letters: 0->'a', 1->'b', etc.
    letters = [chr(ord('a') + i) for i in range(max_idx + 1)]
    
    # Build subscript for each arg
    arg_subs = []
    for spec in specs:
        sub = "".join(letters[i] for i in spec)
        arg_subs.append(sub)
    
    # Build output subscript: axes nsum, nsum+1, ..., max_idx
    out_sub = "".join(letters[i] for i in range(nsum, max_idx + 1))
    
    einsum_str = ",".join(arg_subs) + "->" + out_sub
    return jnp.einsum(einsum_str, *args)


@register_op("call")
def _eval_call(op_str, vals, lib_refs, libs, parent_args):
    lib_idx = lib_refs[0]
    call_args = vals
    lib_body = libs[lib_idx]
    result = eval_body(lib_body, libs, call_args)
    if len(result) == 1:
        return result[0]
    return result
