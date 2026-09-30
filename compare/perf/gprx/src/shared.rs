//! Helpers the modes share: the RBF kernel of a case, column-major points.

use gprx::kernel::{KernelSpec, RbfArdKernel, RbfKernel};

/// The case's RBF kernel: ARD over `lengthscales`, or isotropic at the first.
pub fn rbf_kernel(ard: bool, lengthscales: &[f64]) -> Result<KernelSpec, String> {
    if ard {
        let spec = RbfArdKernel::new(lengthscales).map_err(|e| e.to_string())?;
        Ok(KernelSpec::from(spec))
    } else {
        let ell = lengthscales
            .first()
            .copied()
            .ok_or_else(|| "missing lengthscale".to_string())?;
        Ok(KernelSpec::from(
            RbfKernel::new(ell).map_err(|e| e.to_string())?,
        ))
    }
}

/// Point `index` of column-major `values` (`n × d`).
pub fn point_at(values: &[f64], n: usize, d: usize, index: usize) -> Vec<f64> {
    (0..d).map(|feature| values[feature * n + index]).collect()
}

/// The first `keep` points of column-major `values` (`n × d`), column-major.
pub fn prefix_colmajor(values: &[f64], n: usize, d: usize, keep: usize) -> Vec<f64> {
    (0..d)
        .flat_map(|feature| (0..keep).map(move |i| values[feature * n + i]))
        .collect()
}
