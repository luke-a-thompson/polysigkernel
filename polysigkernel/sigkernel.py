import jax
import jax.numpy as jnp
from functools import partial

from .monomial_approximation_solver import MonomialApproximationSolver
from .monomial_interpolation_solver import MonomialInterpolationSolver
from .utils import add_time_fn, interpolate_fn
from .utils import _check_positive_integer, _check_positive_value
from .config import _SOLVERS, _KERNELS, _INTERPOLATIONS


class SigKernel:
    """
    Signature Kernel class for computing kernel matrices, distances, scoring rules,
    and MMD-based measures on time-series (path) data.
    Implementation from https://arxiv.org/abs/2502.08470.
    """

    def __init__(
        self,
        order: int = 5,
        static_kernel: str = "linear",
        solver: str = "monomial_approx",
        refinement_factor: int = 1,
        scale: float = 1.0,
        s0: float = 0.0,
        t0: float = 0.0,
        S: float = 1.0,
        T: float = 1.0,
        add_time: bool = False,
        interpolation: str = "linear",
        multi_gpu: bool = False,
    ):
        _check_positive_integer(order, "order")
        _check_positive_integer(refinement_factor, "refinement_factor")
        _check_positive_value(scale, "scale")

        if static_kernel not in _KERNELS:
            raise ValueError("Static kernel not implemented.")

        if solver not in _SOLVERS:
            raise ValueError("Solver not implemented.")

        if interpolation not in _INTERPOLATIONS:
            raise ValueError("Interpolation not implemented.")

        if solver == "monomial_approx":
            self.solver = MonomialApproximationSolver
        elif solver == "monomial_interp":
            self.solver = MonomialInterpolationSolver

        self.static_kernel = static_kernel
        self.order = order
        self.refinement_factor = refinement_factor
        self.interpolation = interpolation
        self.multi_gpu = multi_gpu

        self.scale = scale

        self.add_time = add_time
        self.s0 = s0
        self.t0 = t0
        self.S = S
        self.T = T

        # Cache solver instances keyed by (python) scale for repeated block solves.
        # This avoids re-allocating solver utility matrices for every block.
        self._solver_cache: dict[
            float, MonomialApproximationSolver | MonomialInterpolationSolver
        ] = {}

    @partial(jax.jit, static_argnums=(0, 3, 4, 5))
    def kernel_matrix(
        self,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        scale: float | None = None,
        max_batch: int | None = None,
        sym: bool = False,
    ) -> jnp.ndarray:
        """
        Compute the signature kernel matrix between sets of paths X and Y.

        If max_batch is specified and the batch dimension exceeds that limit,
        the computation is split recursively to avoid memory issues.

        Args:
            X (jax.numpy.ndarray): Shape (batch_X, length_X, dim).
            Y (jax.numpy.ndarray): Shape (batch_Y, length_Y, dim).
            max_batch (int, optional): Maximum allowed batch size for a single
                                       kernel computation. Defaults to None
                                       (i.e., no splitting).
            sym (bool, optional): If True, the result is treated as symmetric, and
                                  sub-blocks are computed accordingly. Defaults to False.

        Returns:
            jax.numpy.ndarray: Kernel matrix of shape (batch_X, batch_Y).
        """

        if self.refinement_factor > 1:
            # Refine X and Y via interpolation
            X = interpolate_fn(
                X,
                t_min=self.s0,
                t_max=self.S,
                refinement_factor=self.refinement_factor,
                kind=self.interpolation,
            )
            Y = interpolate_fn(
                Y,
                t_min=self.t0,
                t_max=self.T,
                refinement_factor=self.refinement_factor,
                kind=self.interpolation,
            )

        if scale is None:
            scale = self.scale
        else:
            _check_positive_value(scale, "scale")

        # Optionally add time as an extra channel
        if self.add_time:
            X = add_time_fn(X, t_min=self.s0, t_max=self.S)
            Y = add_time_fn(Y, t_min=self.t0, t_max=self.T)

        batch_X, batch_Y = X.shape[0], Y.shape[0]

        # If no splitting is necessary (or no max_batch is provided):
        if (max_batch is None) or (batch_X <= max_batch and batch_Y <= max_batch):
            return self.solver(
                static_ker=self.static_kernel, scale=scale, order=self.order
            ).solve(X, Y, sym, self.multi_gpu)

        # Case 1: X small enough, Y large
        elif batch_X <= max_batch and batch_Y > max_batch:
            cutoff = int(batch_Y / 2)
            Y1, Y2 = Y[:cutoff], Y[cutoff:]
            K1 = self.kernel_matrix(X, Y1, scale=scale, max_batch=max_batch, sym=False)
            K2 = self.kernel_matrix(X, Y2, scale=scale, max_batch=max_batch, sym=False)
            return jnp.concatenate((K1, K2), axis=1)

        # Case 2: X large, Y small enough
        elif batch_X > max_batch and batch_Y <= max_batch:
            cutoff = int(batch_X / 2)
            X1, X2 = X[:cutoff], X[cutoff:]
            K1 = self.kernel_matrix(X1, Y, scale=scale, max_batch=max_batch, sym=False)
            K2 = self.kernel_matrix(X2, Y, scale=scale, max_batch=max_batch, sym=False)
            return jnp.concatenate((K1, K2), axis=0)

        # Case 3: Both X, Y large
        else:
            cutoff_X, cutoff_Y = int(batch_X / 2), int(batch_Y / 2)
            X1, X2 = X[:cutoff_X], X[cutoff_X:]
            Y1, Y2 = Y[:cutoff_Y], Y[cutoff_Y:]

            # Compute sub-blocks
            K11 = self.kernel_matrix(X1, Y1, scale=scale, max_batch=max_batch, sym=sym)
            K12 = self.kernel_matrix(
                X1, Y2, scale=scale, max_batch=max_batch, sym=False
            )
            K22 = self.kernel_matrix(X2, Y2, scale=scale, max_batch=max_batch, sym=sym)

            if sym:
                K21 = K12.swapaxes(0, 1)
            else:
                K21 = self.kernel_matrix(
                    X2, Y1, scale=scale, max_batch=max_batch, sym=False
                )

            K_top = jnp.concatenate((K11, K12), axis=1)
            K_bottom = jnp.concatenate((K21, K22), axis=1)
            K = jnp.concatenate((K_top, K_bottom), axis=0)
            return K

    ########################################################################################

    def _get_solver_instance(
        self, scale: float
    ) -> MonomialApproximationSolver | MonomialInterpolationSolver:
        cached = self._solver_cache.get(scale)
        if cached is None:
            cached = self.solver(
                static_ker=self.static_kernel, scale=scale, order=self.order
            )
            self._solver_cache[scale] = cached
        return cached

    def _prepare_inputs(
        self,
        X: jax.Array,
        Y: jax.Array,
        scale: float | None,
    ) -> tuple[jax.Array, jax.Array, float]:
        """
        Apply interpolation/time-augmentation exactly once for scalar computations.
        Returns (X_prepared, Y_prepared, scale_float).
        """
        if self.refinement_factor > 1:
            X = interpolate_fn(
                X,
                t_min=self.s0,
                t_max=self.S,
                refinement_factor=self.refinement_factor,
                kind=self.interpolation,
            )
            Y = interpolate_fn(
                Y,
                t_min=self.t0,
                t_max=self.T,
                refinement_factor=self.refinement_factor,
                kind=self.interpolation,
            )

        if scale is None:
            scale_f = float(self.scale)
        else:
            _check_positive_value(scale, "scale")
            scale_f = float(scale)

        if self.add_time:
            X = add_time_fn(X, t_min=self.s0, t_max=self.S)
            Y = add_time_fn(Y, t_min=self.t0, t_max=self.T)

        return X, Y, scale_f

    @staticmethod
    def _split_blocks(Z: jax.Array, max_batch: int) -> list[jax.Array]:
        if max_batch <= 0:
            raise ValueError("max_batch must be a positive integer.")
        return [Z[i : i + max_batch] for i in range(0, Z.shape[0], max_batch)]

    def _kernel_sum_all(
        self,
        X: jax.Array,
        Y: jax.Array,
        *,
        scale: float,
        max_batch: int,
    ) -> tuple[jax.Array, int]:
        """
        Compute sum_{i,j} k(X_i, Y_j) without materializing the full matrix.
        Returns (sum, count) where count == batch_X * batch_Y.
        """
        solver = self._get_solver_instance(scale)
        X_blocks = self._split_blocks(X, max_batch=max_batch)
        Y_blocks = self._split_blocks(Y, max_batch=max_batch)

        total = jnp.array(0.0, dtype=X.dtype)
        count = 0
        for Xb in X_blocks:
            for Yb in Y_blocks:
                K = solver.solve(Xb, Yb, sym=False, multi_gpu=self.multi_gpu)
                total = total + jnp.sum(K)
                count += int(Xb.shape[0]) * int(Yb.shape[0])
        return total, count

    def _kernel_sum_offdiag_symmetric(
        self,
        X: jax.Array,
        *,
        scale: float,
        max_batch: int,
    ) -> tuple[jax.Array, int]:
        """
        Compute sum_{i!=j} k(X_i, X_j) without materializing the full matrix.
        Returns (sum_offdiag, count_offdiag) where count_offdiag == n*(n-1).
        """
        solver = self._get_solver_instance(scale)
        blocks = self._split_blocks(X, max_batch=max_batch)

        total = jnp.array(0.0, dtype=X.dtype)
        count = 0
        for i, Xi in enumerate(blocks):
            # Diagonal block: subtract diagonal entries to get off-diagonal contribution.
            Kii = solver.solve(Xi, Xi, sym=False, multi_gpu=self.multi_gpu)
            total = total + (jnp.sum(Kii) - jnp.sum(jnp.diag(Kii)))
            count += int(Xi.shape[0]) * int(Xi.shape[0] - 1)

            # Upper triangle blocks: compute once, multiply by 2 to account for (i,j) and (j,i).
            for j in range(i + 1, len(blocks)):
                Xj = blocks[j]
                Kij = solver.solve(Xi, Xj, sym=False, multi_gpu=self.multi_gpu)
                s = jnp.sum(Kij)
                total = total + (2.0 * s)
                count += 2 * int(Xi.shape[0]) * int(Xj.shape[0])

        return total, count

    def compute_distance(
        self, X: jax.Array, Y: jax.Array, max_batch: int = 100
    ) -> jax.Array:
        """
        Input:
            - X: jnp array of shape (batch, length_X, dim),
            - Y: jnp array of shape (batch, length_Y, dim)
        Output:
            - scalar distance based on kernel means
        """
        Xp, Yp, scale = self._prepare_inputs(X, Y, scale=None)

        sum_xx, cnt_xx = self._kernel_sum_all(Xp, Xp, scale=scale, max_batch=max_batch)
        sum_yy, cnt_yy = self._kernel_sum_all(Yp, Yp, scale=scale, max_batch=max_batch)
        sum_xy, cnt_xy = self._kernel_sum_all(Xp, Yp, scale=scale, max_batch=max_batch)

        mean_xx = sum_xx / jnp.array(max(cnt_xx, 1), dtype=Xp.dtype)
        mean_yy = sum_yy / jnp.array(max(cnt_yy, 1), dtype=Yp.dtype)
        mean_xy = sum_xy / jnp.array(max(cnt_xy, 1), dtype=Xp.dtype)

        return mean_xx + mean_yy - 2.0 * mean_xy

    def compute_scoring_rule(
        self, X: jax.Array, y: jax.Array, max_batch: int = 100
    ) -> jax.Array:
        """
        Input:
            - X: jnp array of shape (batch, length_X, dim),
            - y: jnp array of shape (1, length_Y, dim)
        Output:
            - signature kernel scoring rule S(X,y) = E[k(X,X)] - 2E[k(X,y)]
        """

        Xp, yp, scale = self._prepare_inputs(X, y, scale=None)

        sum_xx_off, cnt_xx_off = self._kernel_sum_offdiag_symmetric(
            Xp, scale=scale, max_batch=max_batch
        )
        sum_xy, cnt_xy = self._kernel_sum_all(Xp, yp, scale=scale, max_batch=max_batch)

        if cnt_xx_off == 0:
            k_xx_off_mean = jnp.array(0.0, dtype=Xp.dtype)
        else:
            k_xx_off_mean = sum_xx_off / jnp.array(cnt_xx_off, dtype=Xp.dtype)

        k_xy_mean = sum_xy / jnp.array(max(cnt_xy, 1), dtype=Xp.dtype)
        return k_xx_off_mean - 2.0 * k_xy_mean

    def compute_expected_scoring_rule(
        self, X: jax.Array, Y: jax.Array, max_batch: int = 100
    ) -> jax.Array:
        """
        Input:
            - X: jnp array of shape (batch_X, length_X, dim),
            - Y: jnp array of shape (batch_Y, length_Y, dim)
        Output:
            - signature kernel expected scoring rule S(X,Y) = E_Y[S(X,y)]
        """

        Xp, Yp, scale = self._prepare_inputs(X, Y, scale=None)

        sum_xx_off, cnt_xx_off = self._kernel_sum_offdiag_symmetric(
            Xp, scale=scale, max_batch=max_batch
        )
        sum_xy, cnt_xy = self._kernel_sum_all(Xp, Yp, scale=scale, max_batch=max_batch)

        if cnt_xx_off == 0:
            k_xx_off_mean = jnp.array(0.0, dtype=Xp.dtype)
        else:
            k_xx_off_mean = sum_xx_off / jnp.array(cnt_xx_off, dtype=Xp.dtype)

        k_xy_mean = sum_xy / jnp.array(max(cnt_xy, 1), dtype=Xp.dtype)
        return k_xx_off_mean - 2.0 * k_xy_mean

    def compute_mmd(
        self, X: jax.Array, Y: jax.Array, max_batch: int = 100
    ) -> jax.Array:
        """
        Input:
            - X: jnp array of shape (batch_X, length_X, dim),
            - Y: jnp array of shape (batch_Y, length_Y, dim)
        Output:
            - scalar: MMD signature distance between samples X and samples Y
        """

        Xp, Yp, scale = self._prepare_inputs(X, Y, scale=None)

        sum_xx_off, cnt_xx_off = self._kernel_sum_offdiag_symmetric(
            Xp, scale=scale, max_batch=max_batch
        )
        sum_yy_off, cnt_yy_off = self._kernel_sum_offdiag_symmetric(
            Yp, scale=scale, max_batch=max_batch
        )
        sum_xy, cnt_xy = self._kernel_sum_all(Xp, Yp, scale=scale, max_batch=max_batch)

        if cnt_xx_off == 0:
            k_xx_off_mean = jnp.array(0.0, dtype=Xp.dtype)
        else:
            k_xx_off_mean = sum_xx_off / jnp.array(cnt_xx_off, dtype=Xp.dtype)

        if cnt_yy_off == 0:
            k_yy_off_mean = jnp.array(0.0, dtype=Yp.dtype)
        else:
            k_yy_off_mean = sum_yy_off / jnp.array(cnt_yy_off, dtype=Yp.dtype)

        k_xy_mean = sum_xy / jnp.array(max(cnt_xy, 1), dtype=Xp.dtype)
        return k_xx_off_mean + k_yy_off_mean - 2.0 * k_xy_mean


########################################################################################
# Hypothesis test functionality
########################################################################################


def c_alpha(m: int, alpha: float) -> jax.Array:
    return 4.0 * jnp.sqrt(-jnp.log(alpha) / m)


def hypothesis_test(
    y_pred: jax.Array,
    y_test: jax.Array,
    static_kernel: str = "linear",
    confidence_level: float = 0.99,
    **kwargs,
) -> None:
    """
    Statistical test based on MMD distance to determine if
    two sets of paths come from the same distribution.
    """

    order = kwargs.get("order", 3)
    max_batch = kwargs.get("max_batch", 100)
    solver = kwargs.get("solver", "monomial_approx")
    refinement_factor = kwargs.get("refinement_factor", 1)
    interpolation = kwargs.get("interpolation", "linear")
    scale = kwargs.get("scale", 1.0)

    k_sig = SigKernel(
        order,
        static_kernel,
        solver=solver,
        refinement_factor=refinement_factor,
        scale=scale,
        interpolation=interpolation,
    )

    m = max(y_pred.shape[0], y_test.shape[0])

    TU = k_sig.compute_mmd(y_pred, y_test, max_batch=max_batch)

    c = jnp.array(c_alpha(m, confidence_level), dtype=y_pred.dtype)

    if TU > c:
        print(
            f"Hypothesis rejected: distribution are not equal with {confidence_level * 100}% confidence"
        )
    else:
        print(
            f"Hypothesis accepted: distribution are equal with {confidence_level * 100}% confidence"
        )


# ===========================================================================================================
