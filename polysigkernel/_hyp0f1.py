import jax
import jax.numpy as jnp
from jax import lax

#############################################
# Hypergeometric function 0F1 implementation
#############################################

DOUBLE_PRECISION = 1e-15


def _hyp_0f1_serie(a, x):

    safe_a = jnp.where(a == 0, jnp.ones_like(a), a)

    def body(state):
        serie, k, term = state
        serie = serie + term
        term = term * x / (k + 1) / (a + k)
        k = k + 1
        return serie, k, term

    def cond(state):
        serie, k, term = state
        return (k < 250) & (lax.abs(term) / lax.abs(serie) > DOUBLE_PRECISION)

    init = (jnp.asarray(1.0, dtype=x.dtype),
            jnp.asarray(1, dtype=jnp.int32),
            x / safe_a)

    return lax.while_loop(cond, body, init)[0]


@jax.jit
@jnp.vectorize
def hyp0f1(a, x):
    """
    Implements the hypergeometric function 0F1 in jax using lax backend.

    Uses a truncated power series (capped at 250 terms). Accurate for |x| up to
    a few hundred; an asymptotic expansion for large |x| is not implemented.
    When a == 0 the function returns +inf by convention.
    """
    result = _hyp_0f1_serie(a, x)
    index = (a == 0) * 1
    return lax.select_n(index, result, jnp.array(jnp.inf, dtype=x.dtype))
