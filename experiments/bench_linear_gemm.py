import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from polysigkernel.sigkernel import SigKernel


def _make_data(
    key: jax.Array,
    batch_size: int,
    length: int,
    dim: int,
    dtype: jnp.dtype,
) -> jax.Array:
    x0 = 1e-1 * jax.random.normal(key, shape=(batch_size, length, dim), dtype=dtype)
    return x0.cumsum(axis=1)


def _bench_once(
    sk: SigKernel,
    X: jax.Array,
    Y: jax.Array,
    *,
    warmup: int,
    runs: int,
) -> tuple[float, float, float, float]:
    # Compile (untimed)
    out = sk.kernel_matrix(X, Y)
    out.block_until_ready()

    # Warmup (untimed)
    for _ in range(warmup):
        out = sk.kernel_matrix(X, Y)
        out.block_until_ready()

    # Steady-state runtime
    times_s: list[float] = []
    for _ in range(runs):
        t1 = time.perf_counter()
        out = sk.kernel_matrix(X, Y)
        out.block_until_ready()
        times_s.append(time.perf_counter() - t1)

    times_sorted = sorted(times_s)
    mean_s = float(sum(times_sorted) / len(times_sorted))
    median_s = float(times_sorted[len(times_sorted) // 2])
    p90_s = float(times_sorted[int(0.9 * (len(times_sorted) - 1))])
    p10_s = float(times_sorted[int(0.1 * (len(times_sorted) - 1))])
    return mean_s, median_s, p90_s, p10_s


def _set_vectorized_diag(sk: SigKernel, enabled: bool) -> None:
    # SigKernel caches solver instances keyed by scale (python float)
    s = float(sk.scale)
    solver = sk._solver_cache[s]
    setattr(solver, "vectorized_diag_update", bool(enabled))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--dtype", type=str, choices=["float32", "float64"], default="float32"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Benchmark both scalar and vectorized diagonal updates.",
    )
    args = parser.parse_args()

    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
        dtype = jnp.float64
    else:
        dtype = jnp.float32

    key_x = jax.random.PRNGKey(0)
    key_y = jax.random.PRNGKey(1)
    X = _make_data(key_x, args.batch_size, args.length, args.dim, dtype)
    Y = _make_data(key_y, args.batch_size, args.length, args.dim, dtype)

    device = str(X.device)
    print(
        f"device={device} dtype={args.dtype} batch={args.batch_size} length={args.length} dim={args.dim} order={args.order}"
    )
    if args.both:
        # Use separate SigKernel instances so the solver branch is baked per compile.
        sk_scalar = SigKernel(
            order=args.order,
            static_kernel="linear",
            solver="monomial_approx",
            add_time=False,
        )
        _set_vectorized_diag(sk_scalar, False)
        mean_s, median_s, p90_s, p10_s = _bench_once(
            sk_scalar, X, Y, warmup=args.warmup, runs=args.runs
        )
        print(
            f"path=scalar mean_s={mean_s:.6f} median_s={median_s:.6f} p10_s={p10_s:.6f} p90_s={p90_s:.6f}"
        )

        sk_vec = SigKernel(
            order=args.order,
            static_kernel="linear",
            solver="monomial_approx",
            add_time=False,
        )
        _set_vectorized_diag(sk_vec, True)
        mean_s, median_s, p90_s, p10_s = _bench_once(
            sk_vec, X, Y, warmup=args.warmup, runs=args.runs
        )
        print(
            f"path=vectorized mean_s={mean_s:.6f} median_s={median_s:.6f} p10_s={p10_s:.6f} p90_s={p90_s:.6f}"
        )
    else:
        sk = SigKernel(
            order=args.order,
            static_kernel="linear",
            solver="monomial_approx",
            add_time=False,
        )
        _set_vectorized_diag(sk, True)
        mean_s, median_s, p90_s, p10_s = _bench_once(
            sk, X, Y, warmup=args.warmup, runs=args.runs
        )
        print(
            f"mean_s={mean_s:.6f} median_s={median_s:.6f} p10_s={p10_s:.6f} p90_s={p90_s:.6f}"
        )


if __name__ == "__main__":
    main()
