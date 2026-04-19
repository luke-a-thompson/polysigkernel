import jax
import jax.numpy as jnp
from typing import Optional

from .monomial_approximation_solver import MonomialApproximationSolver
from .monomial_interpolation_solver import MonomialInterpolationSolver
from .utils import add_time_fn, interpolate_fn
from .utils import _check_positive_integer, _check_positive_value
from .config import SolverType, StaticKernelType, InterpolationType
from typing import get_args


class SigKernel:
    """
    Signature Kernel class for computing kernel matrices, distances, scoring rules,
    and MMD-based measures on time-series (path) data.
    Implementation from https://arxiv.org/abs/2502.08470.
    """

    def __init__(
        self,
        order: int = 5,
        static_kernel: StaticKernelType = "linear",
        solver: SolverType = "monomial_approx",
        refinement_factor: int = 1,
        scale: float = 1.0,
        s0: float = 0.0,
        t0: float = 0.0,
        S: float = 1.0,
        T: float = 1.0,
        add_time: bool = False,
        interpolation: InterpolationType = "linear",
        multi_gpu: bool = False,
    ):

        _check_positive_integer(order, "order")
        _check_positive_integer(refinement_factor, "refinement_factor")
        _check_positive_value(scale, "scale")

        if static_kernel not in get_args(StaticKernelType):
            raise ValueError("Static kernel not implemented.")

        if solver not in get_args(SolverType):
            raise ValueError("Solver not implemented.")

        if interpolation not in get_args(InterpolationType):
            raise ValueError("Interpolation not implemented.")

        if solver == "monomial_approx":
            self._solver_cls = MonomialApproximationSolver
        elif solver == "monomial_interp":
            self._solver_cls = MonomialInterpolationSolver

        self.static_kernel = static_kernel
        self.order = order
        self.refinement_factor = refinement_factor
        self.interpolation = interpolation
        self.multi_gpu = multi_gpu

        self.scale = scale

        self._solver = self._solver_cls(
            static_ker=static_kernel, scale=scale, order=order
        )

        self.add_time = add_time
        self.s0 = s0
        self.t0 = t0
        self.S = S
        self.T = T

    @staticmethod
    def _batch_slices(batch_size: int, max_batch: Optional[int]) -> list[slice]:
        if (max_batch is None) or (batch_size <= max_batch):
            return [slice(0, batch_size)]

        return [
            slice(start, min(start + max_batch, batch_size))
            for start in range(0, batch_size, max_batch)
        ]

    def _prepare_inputs(
        self,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        scale: Optional[float],
        max_batch: Optional[int],
    ) -> tuple[jnp.ndarray, jnp.ndarray, object]:
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
            scale = self.scale
        else:
            _check_positive_value(scale, "scale")

        if max_batch is not None:
            _check_positive_integer(max_batch, "max_batch")

        if self.add_time:
            X = add_time_fn(X, t_min=self.s0, t_max=self.S)
            Y = add_time_fn(Y, t_min=self.t0, t_max=self.T)

        if scale == self.scale:
            solver = self._solver
        else:
            solver = self._solver_cls(
                static_ker=self.static_kernel, scale=scale, order=self.order
            )

        return X, Y, solver

    def _solve_cross_kernel_block(
        self,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        solver,
    ) -> jax.Array:
        if X.shape[1] < Y.shape[1]:
            return solver.solve(Y, X, False, self.multi_gpu).swapaxes(0, 1)

        return solver.solve(X, Y, False, self.multi_gpu)

    def _cross_kernel_sum(
        self,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        solver,
        max_batch: Optional[int],
    ) -> jax.Array:
        total = jnp.array(0.0, dtype=X.dtype)

        for x_slice in self._batch_slices(X.shape[0], max_batch):
            Xi = X[x_slice]
            for y_slice in self._batch_slices(Y.shape[0], max_batch):
                total = total + jnp.sum(
                    self._solve_cross_kernel_block(Xi, Y[y_slice], solver)
                )

        return total

    def _self_kernel_sums(
        self,
        X: jnp.ndarray,
        solver,
        max_batch: Optional[int],
    ) -> tuple[jax.Array, jax.Array]:
        total = jnp.array(0.0, dtype=X.dtype)
        diag_total = jnp.array(0.0, dtype=X.dtype)
        x_slices = self._batch_slices(X.shape[0], max_batch)

        for i, x_slice in enumerate(x_slices):
            Xi = X[x_slice]
            diag_block = solver.solve(Xi, Xi, True, self.multi_gpu)
            total = total + jnp.sum(diag_block)
            diag_total = diag_total + jnp.sum(jnp.diag(diag_block))

            for y_slice in x_slices[i + 1 :]:
                offdiag_block = solver.solve(Xi, X[y_slice], False, self.multi_gpu)
                total = total + 2.0 * jnp.sum(offdiag_block)

        return total, diag_total

    def kernel_matrix(
        self,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        scale: Optional[float] = None,
        max_batch: Optional[int] = None,
        sym: bool = False,
    ) -> jnp.ndarray:
        """
        Compute the signature kernel matrix between sets of paths X and Y.

        If max_batch is specified and the batch dimension exceeds that limit,
        the computation is split into fixed-size blocks to avoid memory issues
        and excessive recompilation.

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
        X, Y, solver = self._prepare_inputs(X, Y, scale, max_batch)
        batch_X, batch_Y = X.shape[0], Y.shape[0]

        # If no splitting is necessary (or no max_batch is provided):
        if (max_batch is None) or (batch_X <= max_batch and batch_Y <= max_batch):
            if sym:
                return solver.solve(X, Y, True, self.multi_gpu)
            return self._solve_cross_kernel_block(X, Y, solver)

        x_slices = self._batch_slices(batch_X, max_batch)
        y_slices = self._batch_slices(batch_Y, max_batch)
        blocks = [[None for _ in y_slices] for _ in x_slices]

        for i, x_slice in enumerate(x_slices):
            for j, y_slice in enumerate(y_slices):
                if sym and j < i:
                    blocks[i][j] = blocks[j][i].swapaxes(0, 1)
                    continue

                block_sym = bool(sym and i == j)
                if block_sym:
                    blocks[i][j] = solver.solve(
                        X[x_slice], Y[y_slice], True, self.multi_gpu
                    )
                else:
                    blocks[i][j] = self._solve_cross_kernel_block(
                        X[x_slice], Y[y_slice], solver
                    )

        return jnp.concatenate(
            [jnp.concatenate(row_blocks, axis=1) for row_blocks in blocks],
            axis=0,
        )

    ########################################################################################

    def compute_distance(self, X, Y, max_batch=100):
        """
        Input:
            - X: jnp array of shape (batch, length_X, dim),
            - Y: jnp array of shape (batch, length_Y, dim)
        Output:
            - vector ||S(X^i)_T - S(Y^i)_T||^2 of shape (batch,)
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                "compute_distance requires X and Y to have the same batch size."
            )

        K_XX = self.kernel_matrix(X, X, sym=True, max_batch=max_batch)
        K_YY = self.kernel_matrix(Y, Y, sym=True, max_batch=max_batch)
        K_XY = self.kernel_matrix(X, Y, max_batch=max_batch)

        return jnp.diag(K_XX) + jnp.diag(K_YY) - 2.0 * jnp.diag(K_XY)

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
        if X.shape[0] < 2:
            raise ValueError("compute_scoring_rule requires batch size >= 2.")

        X, y, solver = self._prepare_inputs(X, y, scale=None, max_batch=max_batch)
        K_XX_sum, K_XX_diag_sum = self._self_kernel_sums(X, solver, max_batch)
        K_Xy_sum = self._cross_kernel_sum(X, y, solver, max_batch)

        K_XX_m = (K_XX_sum - K_XX_diag_sum) / (X.shape[0] * (X.shape[0] - 1.0))
        K_Xy_m = K_Xy_sum / (X.shape[0] * y.shape[0])

        return K_XX_m - 2.0 * K_Xy_m

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
        if X.shape[0] < 2:
            raise ValueError(
                "compute_expected_scoring_rule requires X batch size >= 2."
            )

        X, Y, solver = self._prepare_inputs(X, Y, scale=None, max_batch=max_batch)
        K_XX_sum, K_XX_diag_sum = self._self_kernel_sums(X, solver, max_batch)
        K_XY_sum = self._cross_kernel_sum(X, Y, solver, max_batch)

        K_XX_m = (K_XX_sum - K_XX_diag_sum) / (X.shape[0] * (X.shape[0] - 1.0))
        K_XY_m = K_XY_sum / (X.shape[0] * Y.shape[0])

        return K_XX_m - 2.0 * K_XY_m

    def compute_expected_scoring_rule_minibatch(
        self,
        X: jax.Array,
        Y: jax.Array,
        key: jax.Array,
        minibatch_size_X: int,
        minibatch_size_Y: Optional[int] = None,
        num_minibatches: int = 10,
        max_batch: int = 100,
    ) -> jax.Array:
        """
        Minibatch estimator for the expected scoring rule S(X,Y) = E_Y[S(X,y)].

        Each iteration subsamples minibatch_size_X paths from X and minibatch_size_Y
        paths from Y, computes the scoring rule on those subsets, then averages over
        num_minibatches iterations. This gives an unbiased estimate while keeping
        memory proportional to the minibatch sizes rather than the full dataset.

        Args:
            X: shape (batch_X, length_X, dim) - forecast paths.
            Y: shape (batch_Y, length_Y, dim) - observation paths.
            key: JAX PRNG key for random subsampling.
            minibatch_size_X: number of X paths per minibatch (must be >= 2).
            minibatch_size_Y: number of Y paths per minibatch (defaults to minibatch_size_X).
            num_minibatches: number of random subsamples to average over.
            max_batch: max batch size passed to the kernel solver.

        Returns:
            scalar: unbiased minibatch estimate of the expected scoring rule.
        """
        if minibatch_size_X < 2:
            raise ValueError("minibatch_size_X must be >= 2.")
        if minibatch_size_Y is None:
            minibatch_size_Y = minibatch_size_X

        X, Y, solver = self._prepare_inputs(X, Y, scale=None, max_batch=max_batch)

        n_X, n_Y = X.shape[0], Y.shape[0]
        replace_X = minibatch_size_X > n_X
        replace_Y = minibatch_size_Y > n_Y

        total = jnp.array(0.0, dtype=X.dtype)
        for _ in range(num_minibatches):
            key, key_x, key_y = jax.random.split(key, 3)
            x_idx = jax.random.choice(
                key_x, n_X, shape=(minibatch_size_X,), replace=replace_X
            )
            y_idx = jax.random.choice(
                key_y, n_Y, shape=(minibatch_size_Y,), replace=replace_Y
            )

            X_mb = X[x_idx]
            Y_mb = Y[y_idx]

            K_XX_sum, K_XX_diag_sum = self._self_kernel_sums(X_mb, solver, max_batch)
            K_XY_sum = self._cross_kernel_sum(X_mb, Y_mb, solver, max_batch)

            K_XX_m = (K_XX_sum - K_XX_diag_sum) / (
                minibatch_size_X * (minibatch_size_X - 1.0)
            )
            K_XY_m = K_XY_sum / (minibatch_size_X * minibatch_size_Y)

            total = total + K_XX_m - 2.0 * K_XY_m

        return total / num_minibatches

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
        if X.shape[0] < 2 or Y.shape[0] < 2:
            raise ValueError("compute_mmd requires batch sizes >= 2 for X and Y.")

        X, Y, solver = self._prepare_inputs(X, Y, scale=None, max_batch=max_batch)
        K_XX_sum, K_XX_diag_sum = self._self_kernel_sums(X, solver, max_batch)
        K_YY_sum, K_YY_diag_sum = self._self_kernel_sums(Y, solver, max_batch)
        K_XY_sum = self._cross_kernel_sum(X, Y, solver, max_batch)

        K_XX_m = (K_XX_sum - K_XX_diag_sum) / (X.shape[0] * (X.shape[0] - 1.0))
        K_YY_m = (K_YY_sum - K_YY_diag_sum) / (Y.shape[0] * (Y.shape[0] - 1.0))
        K_XY_m = K_XY_sum / (X.shape[0] * Y.shape[0])

        return K_XX_m + K_YY_m - 2.0 * K_XY_m


########################################################################################
# Hypothesis test functionality
########################################################################################


def c_alpha(m: int, alpha: float) -> jax.Array:
    return 4.0 * jnp.sqrt(-jnp.log(alpha) / m)


def hypothesis_test(
    y_pred: jax.Array,
    y_test: jax.Array,
    static_kernel: StaticKernelType = "linear",
    confidence_level: float = 0.99,
    **kwargs,
):
    """
    Statistical test based on MMD distance to determine if
    two sets of paths come from the same distribution.
    """

    order = kwargs.get("order", 5)
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

    rejected = bool(TU > c)
    if rejected:
        print(
            f"Hypothesis rejected: distributions are not equal with {confidence_level * 100}% confidence"
        )
    else:
        print(
            f"Hypothesis accepted: distributions are equal with {confidence_level * 100}% confidence"
        )
    return rejected


# ===========================================================================================================
