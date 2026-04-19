import jax.numpy as jnp


def diag_axis_masks(p: int, length_X: int, length_Y: int):
    diag_length_solution = 2 * (length_X - 1)
    diag_length_data = length_X - 1

    diag_axis_solution = jnp.arange(diag_length_solution)
    diag_axis_data = jnp.arange(diag_length_data)

    start_row_solution = jnp.where(
        p == length_X + length_Y - 3,
        jnp.maximum(0, 2 * (p - length_Y + 1)),
        jnp.maximum(0, 2 * (p - length_Y + 2)),
    )
    end_row_solution = jnp.minimum(diag_length_solution, 2 * p + 2)

    start_row_data = jnp.maximum(0, p - length_Y + 1)
    end_row_data = jnp.minimum(diag_length_data, p)

    mask_solution = jnp.where(
        (diag_axis_solution >= start_row_solution)
        & (diag_axis_solution < end_row_solution),
        diag_axis_solution,
        -1,
    )
    mask_data = jnp.where(
        (diag_axis_data >= start_row_data) & (diag_axis_data < end_row_data),
        diag_axis_data,
        -1,
    )

    return jnp.where(
        p == length_X + length_Y - 3, mask_solution.at[-2:].add(1), mask_solution
    ), mask_data


def get_idx(k: int):
    return jnp.where(k % 2 == 0, k - 2, k), k - 1, (k + 1) // 2 - 1
