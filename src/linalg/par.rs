//! Describes the faer worker caps shared by every factorization and solve.

use faer::Par;

/// Caps a square `n×n` faer kernel at `min(pool, n/64, n²/16384)`.
///
/// For square `n` this matches `min(pool, n/64)` used by Cholesky and the
/// `W` n-RHS solve. See [`faer_par_dims`] and
/// `docs/adr/0001-faer-parallel-degree.md`.
#[inline]
pub(crate) fn faer_par(n: usize) -> Par {
    faer_par_dims(n, n)
}

/// Caps faer workers by `n/64`, `n·k/16384`, and `k/12` (`k` = RHS columns).
///
/// Predict `L⁻¹ k_*` is `n×m` with `m = 100` in `compare/perf`. Using
/// [`faer_par`] (`k = n`) starts 16 workers; n=1024 jumps 1.5–22 ms and
/// n=4096 can spike above 100 ms. Kernel Rayon is unchanged.
#[inline]
pub(crate) fn faer_par_dims(nrows: usize, ncols: usize) -> Par {
    Par::rayon(faer_degree(nrows, ncols, rayon::current_num_threads()))
}

/// Worker cap: `min(pool, n/64, n·k/16384, max(1, k/12))`.
pub(crate) fn faer_degree(nrows: usize, ncols: usize, pool: usize) -> usize {
    let pool = pool.max(1);
    let by_n = (nrows / 64).max(1);
    let by_work = (nrows.saturating_mul(ncols) / 16_384).max(1);
    let by_rhs = (ncols / 12).max(1);
    pool.min(by_n).min(by_work).min(by_rhs)
}

#[cfg(test)]
mod tests {
    use super::faer_degree;

    #[test]
    fn faer_degree_keeps_square_n_over_64() {
        assert_eq!(faer_degree(256, 256, 16), 4);
        assert_eq!(faer_degree(1024, 1024, 16), 16);
        assert_eq!(faer_degree(4096, 4096, 16), 16);
    }

    #[test]
    fn faer_degree_caps_skinny_predict_rhs() {
        assert_eq!(faer_degree(256, 100, 16), 1);
        assert_eq!(faer_degree(1024, 100, 16), 6);
        assert_eq!(faer_degree(4096, 100, 16), 8);
        assert_eq!(faer_degree(1024, 1, 16), 1);
    }
}
