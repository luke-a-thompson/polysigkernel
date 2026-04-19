import jax
import jax.numpy as jnp
from functools import partial
from .config import StaticKernelType
from .static_kernels import linear_kernel, rbf_kernel
from ._solver_utils import diag_axis_masks, get_idx


class MonomialApproximationSolver:
    """
    Solver for signature kernel PDE that uses monomial approximation.

    Args:
        static_ker (str, optional) : Type of static kernel to use. Supported values:
                                     'linear' or 'rbf'. Defaults to 'linear'.
        scale (float, optional)    : Scaling parameter for the static kernel. Defaults to 1.0.
        order (int, optional)      : Monomial expansion order. Defaults to 4.
    """

    def __init__(
        self,
        static_ker: StaticKernelType = "linear",
        scale: float = 1.0,
        order: int = 4,
    ):

        self.order = order
        self.static_ker = static_ker
        self.scale = scale

        # Precompute utility matrices (mat1 and mat2) used for monomial-based updates.
        self.mat1, self.mat2 = self._compute_utils(order)

        # Initialize the desired static kernel (linear or RBF) with the given scale.
        if static_ker == "linear":
            self.static_kernel = lambda x2, x1, y2, y1: linear_kernel(
                x2, x1, y2, y1, scale
            )

        elif static_ker == "rbf":
            self.static_kernel = lambda x2, x1, y2, y1: rbf_kernel(
                x2, x1, y2, y1, scale
            )

    ########################################################################
    # Utility functions
    ########################################################################

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _compute_utils(order: int):
        """
        Compute utility matrices for monomial expansions up to a given order.
        """
        n_range = jnp.arange(1, order + 1)[:, None]
        k_range = jnp.arange(order + 1)

        factorials = jnp.cumprod(jnp.arange(1, 2 * order + 1))
        factorials = jnp.concatenate([jnp.array([1.0]), factorials])

        k_over_n = factorials[k_range[None, :]] / factorials[n_range[: order + 1]]

        mat1 = jnp.where(
            jnp.tril(jnp.ones((order - 1, order - 1))),
            k_over_n[1:, 1:-1] / jnp.tril(factorials[n_range[1:] - k_range[1:-1]]),
            0.0,
        )
        mat2 = k_over_n / factorials[n_range + k_range[None, :]]

        return mat1, mat2

    @staticmethod
    def _initial_conditions(order: int, dtype):
        ic = jnp.zeros(shape=(order + 1), dtype=dtype)
        return ic.at[0].set(1.0)

    @staticmethod
    def _active_diag_window(p: int, length_X: int, length_Y: int):
        diag_length = 2 * (length_X - 1)
        final_diag = length_X + length_Y - 3
        is_final = p == final_diag

        first_k = jnp.where(
            is_final,
            diag_length - 1,
            jnp.maximum(0, 2 * (p - length_Y + 2)),
        )
        active_len = jnp.where(
            is_final,
            2,
            jnp.minimum(diag_length, 2 * p + 2) - first_k,
        )
        return first_k, active_len

    def _direct_mat1_update(self, prev_bd: jnp.ndarray, ker: jax.Array) -> jnp.ndarray:
        prev_tail = prev_bd[..., 1:-1]

        if self.order <= 1:
            return jnp.empty(prev_tail.shape[:-1] + (0,), dtype=prev_bd.dtype)

        tail_update = []
        for row in range(self.order - 1):
            coeffs = self.mat1[row, : row + 1] * prev_tail[..., : row + 1]
            acc = coeffs[..., 0]
            for coeff_idx in range(1, row + 1):
                acc = acc * ker + coeffs[..., coeff_idx]
            tail_update.append(acc * ker)

        return jnp.stack(tail_update, axis=-1)

    ########################################################################
    # Diagonal updates
    ########################################################################

    @partial(jax.jit, static_argnums=(0, 5))
    def _get_diag_data_generic(
        self,
        p: int,
        diag_axis_masked: jnp.ndarray,
        X: jnp.ndarray,
        Y: jnp.ndarray,
        sym: bool = False,
    ) -> jnp.ndarray:
        valid_mask = diag_axis_masked != -1
        x_idx = jnp.where(valid_mask, diag_axis_masked, 0)
        y_idx = jnp.where(valid_mask, p - diag_axis_masked - 1, 0)

        x2 = jnp.take(X, x_idx + 1, axis=1)[:, None, :, :]
        x1 = jnp.take(X, x_idx, axis=1)[:, None, :, :]
        y2 = jnp.take(Y, y_idx + 1, axis=1)[None, :, :, :]
        y1 = jnp.take(Y, y_idx, axis=1)[None, :, :, :]

        diag_data = self.static_kernel(x2, x1, y2, y1)
        diag_data = diag_data * valid_mask[None, None, :].astype(X.dtype)

        if sym:
            lower_tri_mask = (
                jnp.arange(X.shape[0])[:, None] >= jnp.arange(Y.shape[0])[None, :]
            )
            diag_data = diag_data * lower_tri_mask[..., None].astype(X.dtype)

        return diag_data

    @partial(jax.jit, static_argnums=(0, 5))
    def _get_diag_data_linear(
        self,
        p: int,
        diag_axis_masked: jnp.ndarray,
        dX: jnp.ndarray,
        dY: jnp.ndarray,
        sym: bool = False,
    ) -> jnp.ndarray:
        valid_mask = diag_axis_masked != -1
        x_idx = jnp.where(valid_mask, diag_axis_masked, 0)
        y_idx = jnp.where(valid_mask, p - diag_axis_masked - 1, 0)

        dX_diag = jnp.take(dX, x_idx, axis=1)
        dY_diag = jnp.take(dY, y_idx, axis=1)

        diag_data = (self.scale**2) * jnp.einsum("xkc,ykc->xyk", dX_diag, dY_diag)
        diag_data = diag_data * valid_mask[None, None, :].astype(dX.dtype)

        if sym:
            lower_tri_mask = (
                jnp.arange(dX.shape[0])[:, None] >= jnp.arange(dY.shape[0])[None, :]
            )
            diag_data = diag_data * lower_tri_mask[..., None].astype(dX.dtype)

        return diag_data

    ########################################################################
    # Solution updates
    ########################################################################

    @partial(jax.jit, static_argnums=(0, 8))
    def _solution_diag_update_active(
        self,
        p: int,
        active_k: jnp.ndarray,
        active_len: jax.Array,
        prev_first_k: jax.Array,
        prev_active_len: jax.Array,
        prev_diag_solution: jnp.ndarray,
        diag_data: jnp.ndarray,
        sym: bool = False,
    ):
        """
        Given the data kernel evaluations for the current diagonal, update the solution
        along that diagonal.
        """
        mat2 = self.mat2

        ic = self._initial_conditions(self.order, diag_data.dtype)
        max_active_len = active_k.shape[0]
        active_mask = jnp.arange(max_active_len) < active_len

        idx1, idx2, idx_data = get_idx(active_k)

        prev_rel_idx1 = idx1 - prev_first_k
        prev_rel_idx2 = idx2 - prev_first_k
        prev_mask1 = (
            active_mask & (prev_rel_idx1 >= 0) & (prev_rel_idx1 < prev_active_len)
        )
        prev_mask2 = (
            active_mask & (prev_rel_idx2 >= 0) & (prev_rel_idx2 < prev_active_len)
        )

        safe_prev_idx1 = jnp.clip(prev_rel_idx1, 0, max_active_len - 1)
        safe_prev_idx2 = jnp.clip(prev_rel_idx2, 0, max_active_len - 1)
        safe_data_idx = jnp.clip(idx_data, 0, diag_data.shape[2] - 1)

        prev_bd = jnp.take(prev_diag_solution, safe_prev_idx1, axis=2)
        prev_bd_opposite = jnp.take(prev_diag_solution, safe_prev_idx2, axis=2)
        prev_bd = prev_bd * prev_mask1[None, None, :, None].astype(
            prev_diag_solution.dtype
        )
        prev_bd_opposite = prev_bd_opposite * prev_mask2[None, None, :, None].astype(
            prev_diag_solution.dtype
        )

        ker = jnp.take(diag_data, safe_data_idx, axis=2)
        ker = ker * active_mask[None, None, :].astype(diag_data.dtype)
        ker_powers = jnp.power(
            ker[..., None], jnp.arange(1, self.order + 1, dtype=diag_data.dtype)
        )

        mat2_update = jnp.einsum("ab,xykb->xyka", mat2, prev_bd_opposite)

        new_bc = prev_bd.at[..., 0].set(jnp.sum(prev_bd_opposite, axis=-1))
        new_bc = new_bc.at[..., 1:].add(mat2_update * ker_powers)
        new_bc = new_bc.at[..., 2:].add(self._direct_mat1_update(prev_bd, ker))

        boundary_mask = active_mask & ((active_k == 0) | (active_k == 2 * p + 1))
        output = jnp.where(boundary_mask[None, None, :, None], ic, new_bc)
        output = output * active_mask[None, None, :, None].astype(diag_data.dtype)

        if sym:
            lower_tri_mask = (
                jnp.arange(diag_data.shape[0])[:, None]
                >= jnp.arange(diag_data.shape[1])[None, :]
            )
            output = output * lower_tri_mask[..., None, None].astype(diag_data.dtype)

        return output

    ########################################################################
    # Main solvers
    ########################################################################

    @partial(jax.jit, static_argnums=(0, 3))
    def _solve(self, X: jnp.ndarray, Y: jnp.ndarray, sym: bool = False) -> jnp.ndarray:
        """
        The core solver method, which iterates over diagonals.

        Args:
            X (jax.numpy.ndarray): 3D array with shape (batch_X, length_X, dim_X).
            Y (jax.numpy.ndarray): 3D array with shape (batch_Y, length_Y, dim_Y).
            sym (bool, optional): If True, enforce symmetry by only updating i >= j.
                                  The final result is combined symmetrically. Defaults to False.

        Returns:
            jax.numpy.ndarray: A 2D solution array of shape (batch_X, batch_Y) containing
                               the final aggregated results of the monomial expansions.
        """
        # Problem sizes
        batch_X, batch_Y = X.shape[0], Y.shape[0]
        length_X, length_Y = X.shape[1], Y.shape[1]
        diag_iterations = length_X + length_Y - 3
        max_active_len = 2 * min(length_X - 1, length_Y - 1)
        dX = X[:, 1:] - X[:, :-1] if self.static_ker == "linear" else None
        dY = Y[:, 1:] - Y[:, :-1] if self.static_ker == "linear" else None

        diag_solution_minus1 = (
            jnp.zeros(
                shape=(batch_X, batch_Y, max_active_len, self.order + 1),
                dtype=X.dtype,
            )
            .at[..., 0]
            .set(1.0)
        )
        init_carry = (
            jnp.array(0, dtype=jnp.int32),
            jnp.array(max_active_len, dtype=jnp.int32),
            diag_solution_minus1,
        )

        def _loop(p, carry):

            prev_first_k, prev_active_len, diag_solution_minus1 = carry
            first_k, active_len = self._active_diag_window(p, length_X, length_Y)
            active_k = first_k + jnp.arange(max_active_len, dtype=jnp.int32)

            _, diag_axis_mask_data = diag_axis_masks(p, length_X, length_Y)
            if self.static_ker == "linear":
                diag_data = self._get_diag_data_linear(
                    p, diag_axis_mask_data, dX, dY, sym
                )
            else:
                diag_data = self._get_diag_data_generic(
                    p, diag_axis_mask_data, X, Y, sym
                )

            diag_solution = self._solution_diag_update_active(
                p,
                active_k,
                active_len,
                prev_first_k,
                prev_active_len,
                diag_solution_minus1,
                diag_data,
                sym,
            )

            return first_k, active_len, diag_solution

        _, final_active_len, diag_solutions = jax.lax.fori_loop(
            1, diag_iterations + 1, _loop, init_carry
        )
        final_mask = (jnp.arange(max_active_len) < final_active_len).astype(X.dtype)
        solution = jnp.sum(
            jnp.sum(diag_solutions, axis=-1) * final_mask[None, None, :],
            axis=-1,
        ) / final_active_len.astype(X.dtype)

        if sym:
            return solution + solution.swapaxes(0, 1) - jnp.diag(jnp.diag(solution))
        else:
            return solution

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def solve(
        self, X: jnp.ndarray, Y: jnp.ndarray, sym: bool = False, multi_gpu: bool = False
    ) -> jnp.ndarray:
        """
        Allows for multi-GPU parallelisation of the solver.
        """

        # Determine the number of GPUs (if any)
        try:
            num_gpus = len(jax.devices("gpu"))
        except RuntimeError:
            num_gpus = 0

        # If only one GPU or no GPU is available, or multi_gpu is False,
        # just run the single-device solver
        if (num_gpus <= 1) or (not multi_gpu):
            return self._solve(X, Y, sym)

        # Find the largest integer "num_parallel" <= "total" that is divisible by num_gpus
        total = X.shape[0]
        num_parallel = (total // num_gpus) * num_gpus
        remaining = total - num_parallel

        # Parallise solver across GPUs. Close over sym so the static arg survives pmap.
        def solve_fn(X_shard, Y_shared):
            return self._solve(X_shard, Y_shared, sym)

        X_parallel = X[:num_parallel]
        X_sub_tensors = jnp.stack(jnp.array_split(X_parallel, num_gpus))
        Z_sub_tensors = jax.pmap(solve_fn, in_axes=(0, None))(X_sub_tensors, Y)
        Z = jnp.concatenate(Z_sub_tensors, axis=0)

        # If all the data has been used just return it
        if remaining == 0:
            return Z

        # Otherwise perform the final computation on a single GPU and concatenate with the rest
        X_remainder = X[num_parallel:]
        Z_remainder = self._solve(X_remainder, Y, sym)

        return jnp.concatenate([Z, Z_remainder], axis=0)
