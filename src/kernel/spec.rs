//! Declaration-layer kernel tree: leaves, sums, and products.

use crate::error::GprError;
use crate::kernel::{
    ConstantKernel, LinearKernel, MaternArdKernel, MaternKernel, RbfArdKernel, RbfKernel,
    WhiteKernel,
};
use std::ops::{Add, Mul};

/// Maps a flat optimizer index to a leaf-local parameter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParameterBinding {
    /// Index in the concatenated kernel parameter vector.
    pub index: usize,
    /// Leaf index in depth-first, left-to-right order.
    pub leaf_id: usize,
    /// Parameter index inside that leaf.
    pub local_index: usize,
}

/// User-facing kernel expression. Parameters stay `f64` until compile.
///
/// Built-in leaves are stored directly. Sum and product nest until
/// [`Self::compile`] flattens associative chains into [`super::CompiledKernel`].
///
/// # Examples
///
/// ```rust
/// use gprx::kernel::{KernelSpec, RbfKernel};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let spec = KernelSpec::from(RbfKernel::new(1.0)?)
///     + KernelSpec::from(RbfKernel::new(2.0)?);
/// assert_eq!(spec.num_params(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum KernelSpec {
    /// Isotropic RBF leaf.
    Rbf(RbfKernel),
    /// ARD RBF leaf (`θ_d = log(ℓ_d)`).
    RbfArd(RbfArdKernel),
    /// Isotropic Matérn leaf (`ν = 1/2`, `3/2`, or `5/2`).
    Matern(MaternKernel),
    /// ARD Matérn leaf (`θ_d = log(ℓ_d)`).
    MaternArd(MaternArdKernel),
    /// Constant leaf `k = c`.
    Constant(ConstantKernel),
    /// Linear leaf `k = σ² xᵀ x'`.
    Linear(LinearKernel),
    /// White (nugget) leaf.
    White(WhiteKernel),
    /// `k = k_left + k_right`.
    Sum(Box<KernelSpec>, Box<KernelSpec>),
    /// `k = k_left * k_right` (Hadamard product).
    Product(Box<KernelSpec>, Box<KernelSpec>),
}

impl From<RbfKernel> for KernelSpec {
    fn from(kernel: RbfKernel) -> Self {
        Self::Rbf(kernel)
    }
}

impl From<RbfArdKernel> for KernelSpec {
    fn from(kernel: RbfArdKernel) -> Self {
        Self::RbfArd(kernel)
    }
}

impl From<MaternKernel> for KernelSpec {
    fn from(kernel: MaternKernel) -> Self {
        Self::Matern(kernel)
    }
}

impl From<MaternArdKernel> for KernelSpec {
    fn from(kernel: MaternArdKernel) -> Self {
        Self::MaternArd(kernel)
    }
}

impl From<ConstantKernel> for KernelSpec {
    fn from(kernel: ConstantKernel) -> Self {
        Self::Constant(kernel)
    }
}

impl From<LinearKernel> for KernelSpec {
    fn from(kernel: LinearKernel) -> Self {
        Self::Linear(kernel)
    }
}

impl From<WhiteKernel> for KernelSpec {
    fn from(kernel: WhiteKernel) -> Self {
        Self::White(kernel)
    }
}

impl Add for KernelSpec {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::Sum(Box::new(self), Box::new(rhs))
    }
}

impl Mul for KernelSpec {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::Product(Box::new(self), Box::new(rhs))
    }
}

impl KernelSpec {
    /// Returns the number of flattened kernel parameters.
    pub fn num_params(&self) -> usize {
        match self {
            Self::Rbf(leaf) => leaf.num_params(),
            Self::RbfArd(leaf) => leaf.num_params(),
            Self::Matern(leaf) => leaf.num_params(),
            Self::MaternArd(leaf) => leaf.num_params(),
            Self::Constant(leaf) => leaf.num_params(),
            Self::Linear(leaf) => leaf.num_params(),
            Self::White(leaf) => leaf.num_params(),
            Self::Sum(left, right) | Self::Product(left, right) => {
                left.num_params() + right.num_params()
            }
        }
    }

    /// Writes flattened `θ` in depth-first, left-to-right leaf order.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `out` is the wrong length.
    pub fn get_params(&self, out: &mut [f64]) -> Result<(), GprError> {
        require_len(out.len(), self.num_params())?;
        let mut offset = 0;
        self.write_params(out, &mut offset);
        Ok(())
    }

    /// Replaces flattened `θ`. All leaves are updated or none are.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::InvalidHyperparameter`] if `params` is the wrong
    /// length or a leaf rejects its slice.
    pub fn set_params(&mut self, params: &[f64]) -> Result<(), GprError> {
        require_len(params.len(), self.num_params())?;
        let mut next = self.clone();
        let mut offset = 0;
        next.apply_params(params, &mut offset)?;
        *self = next;
        Ok(())
    }

    /// Returns the mapping from flat indices to leaves.
    pub fn parameter_bindings(&self) -> Vec<ParameterBinding> {
        let mut out = Vec::new();
        let mut index = 0;
        let mut leaf_id = 0;
        self.collect_bindings(&mut out, &mut index, &mut leaf_id);
        out
    }

    /// Compiles this tree. Associative sums and products become a single list.
    ///
    /// Mixed operators keep their grouping: `(A + B) * C` is a product of a
    /// flattened sum and `C`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use gprx::kernel::{CompiledKernel, KernelSpec, RbfKernel};
    ///
    /// # fn main() -> Result<(), gprx::GprError> {
    /// let spec = (KernelSpec::from(RbfKernel::new(1.0)?)
    ///     + KernelSpec::from(RbfKernel::new(2.0)?))
    ///     + KernelSpec::from(RbfKernel::new(3.0)?);
    /// assert!(matches!(spec.compile(), CompiledKernel::Sum(terms) if terms.len() == 3));
    /// # Ok(())
    /// # }
    /// ```
    pub fn compile(&self) -> crate::kernel::CompiledKernel {
        crate::kernel::CompiledKernel::from_spec(self)
    }

    fn write_params(&self, out: &mut [f64], offset: &mut usize) {
        match self {
            Self::Rbf(leaf) => {
                out[*offset] = leaf.log_lengthscale();
                *offset += 1;
            }
            Self::RbfArd(leaf) => {
                let n = leaf.num_params();
                out[*offset..*offset + n].copy_from_slice(leaf.log_lengthscales());
                *offset += n;
            }
            Self::Matern(leaf) => {
                out[*offset] = leaf.log_lengthscale();
                *offset += 1;
            }
            Self::MaternArd(leaf) => {
                let n = leaf.num_params();
                out[*offset..*offset + n].copy_from_slice(leaf.log_lengthscales());
                *offset += n;
            }
            Self::Constant(leaf) => {
                out[*offset] = leaf.log_constant();
                *offset += 1;
            }
            Self::Linear(leaf) => {
                out[*offset] = leaf.log_variance();
                *offset += 1;
            }
            Self::White(leaf) => {
                out[*offset] = leaf.log_variance();
                *offset += 1;
            }
            Self::Sum(left, right) | Self::Product(left, right) => {
                left.write_params(out, offset);
                right.write_params(out, offset);
            }
        }
    }

    fn apply_params(&mut self, params: &[f64], offset: &mut usize) -> Result<(), GprError> {
        match self {
            Self::Rbf(leaf) => {
                let n = leaf.num_params();
                leaf.set_params(&params[*offset..*offset + n])?;
                *offset += n;
                Ok(())
            }
            Self::RbfArd(leaf) => {
                let n = leaf.num_params();
                leaf.set_params(&params[*offset..*offset + n])?;
                *offset += n;
                Ok(())
            }
            Self::Matern(leaf) => {
                let n = leaf.num_params();
                leaf.set_params(&params[*offset..*offset + n])?;
                *offset += n;
                Ok(())
            }
            Self::MaternArd(leaf) => {
                let n = leaf.num_params();
                leaf.set_params(&params[*offset..*offset + n])?;
                *offset += n;
                Ok(())
            }
            Self::Constant(leaf) => {
                let n = leaf.num_params();
                leaf.set_params(&params[*offset..*offset + n])?;
                *offset += n;
                Ok(())
            }
            Self::Linear(leaf) => {
                let n = leaf.num_params();
                leaf.set_params(&params[*offset..*offset + n])?;
                *offset += n;
                Ok(())
            }
            Self::White(leaf) => {
                let n = leaf.num_params();
                leaf.set_params(&params[*offset..*offset + n])?;
                *offset += n;
                Ok(())
            }
            Self::Sum(left, right) | Self::Product(left, right) => {
                left.apply_params(params, offset)?;
                right.apply_params(params, offset)
            }
        }
    }

    fn collect_bindings(
        &self,
        out: &mut Vec<ParameterBinding>,
        index: &mut usize,
        leaf_id: &mut usize,
    ) {
        match self {
            Self::Rbf(leaf) => {
                push_leaf_bindings(out, index, leaf_id, leaf.num_params());
            }
            Self::RbfArd(leaf) => {
                push_leaf_bindings(out, index, leaf_id, leaf.num_params());
            }
            Self::Matern(leaf) => {
                push_leaf_bindings(out, index, leaf_id, leaf.num_params());
            }
            Self::MaternArd(leaf) => {
                push_leaf_bindings(out, index, leaf_id, leaf.num_params());
            }
            Self::Constant(leaf) => {
                push_leaf_bindings(out, index, leaf_id, leaf.num_params());
            }
            Self::Linear(leaf) => {
                push_leaf_bindings(out, index, leaf_id, leaf.num_params());
            }
            Self::White(leaf) => {
                push_leaf_bindings(out, index, leaf_id, leaf.num_params());
            }
            Self::Sum(left, right) | Self::Product(left, right) => {
                left.collect_bindings(out, index, leaf_id);
                right.collect_bindings(out, index, leaf_id);
            }
        }
    }
}

fn push_leaf_bindings(
    out: &mut Vec<ParameterBinding>,
    index: &mut usize,
    leaf_id: &mut usize,
    n_params: usize,
) {
    let id = *leaf_id;
    *leaf_id += 1;
    for local_index in 0..n_params {
        out.push(ParameterBinding {
            index: *index,
            leaf_id: id,
            local_index,
        });
        *index += 1;
    }
}

fn require_len(actual: usize, expected: usize) -> Result<(), GprError> {
    if actual == expected {
        Ok(())
    } else {
        Err(GprError::InvalidHyperparameter {
            reason: format!("expected {expected} kernel parameters, got {actual}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::KernelSpec;
    use crate::kernel::{
        ConstantKernel, MaternArdKernel, MaternKernel, MaternNu, RbfArdKernel, RbfKernel,
        WhiteKernel,
    };

    const TOL: f64 = 1e-12;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOL * scale,
            "actual={actual}, expected={expected}"
        );
    }

    fn rbf(ell: f64) -> KernelSpec {
        KernelSpec::from(RbfKernel::new(ell).expect("valid"))
    }

    #[test]
    fn sum_flattens_params_left_to_right() {
        let mut spec = (rbf(1.0) + rbf(2.0)) + rbf(3.0);
        assert_eq!(spec.num_params(), 3);
        let mut params = [0.0; 3];
        spec.get_params(&mut params).expect("len 3");
        assert_close(params[0], 1.0_f64.ln());
        assert_close(params[1], 2.0_f64.ln());
        assert_close(params[2], 3.0_f64.ln());
        params[1] = 4.0_f64.ln();
        spec.set_params(&params).expect("len 3");
        spec.get_params(&mut params).expect("len 3");
        assert_close(params[1], 4.0_f64.ln());
        match spec.compile() {
            crate::kernel::CompiledKernel::Sum(terms) => assert_eq!(terms.len(), 3),
            other => panic!("expected flattened sum, got {other:?}"),
        }
    }

    #[test]
    fn product_keeps_sum_nested() {
        let spec = rbf(1.0) * (rbf(2.0) + rbf(3.0));
        match spec.compile() {
            crate::kernel::CompiledKernel::Product(factors) => {
                assert_eq!(factors.len(), 2);
                assert!(matches!(factors[1], crate::kernel::CompiledKernel::Sum(_)));
            }
            other => panic!("expected product, got {other:?}"),
        }
    }

    #[test]
    fn product_flattens_three_factors() {
        let spec = (rbf(1.0) * rbf(2.0)) * rbf(3.0);
        match spec.compile() {
            crate::kernel::CompiledKernel::Product(factors) => assert_eq!(factors.len(), 3),
            other => panic!("expected flattened product, got {other:?}"),
        }
    }

    #[test]
    fn set_params_is_atomic() {
        let mut spec = rbf(1.0) + rbf(2.0);
        let before = spec.clone();
        assert!(spec.set_params(&[0.0, f64::INFINITY]).is_err());
        assert_eq!(spec, before);
    }

    #[test]
    fn bindings_follow_leaves() {
        let spec = rbf(1.0) + rbf(2.0);
        let b = spec.parameter_bindings();
        assert_eq!(b.len(), 2);
        assert_eq!(b[0].leaf_id, 0);
        assert_eq!(b[1].leaf_id, 1);
        assert_eq!(b[0].local_index, 0);
        assert_eq!(b[1].index, 1);
    }

    #[test]
    fn ard_flattens_per_dimension_params() {
        let mut spec = KernelSpec::from(RbfArdKernel::new(&[1.0, 2.0]).expect("valid"));
        assert_eq!(spec.num_params(), 2);
        let mut params = [0.0; 2];
        spec.get_params(&mut params).expect("len 2");
        assert_close(params[0], 1.0_f64.ln());
        assert_close(params[1], 2.0_f64.ln());
        params[1] = 3.0_f64.ln();
        spec.set_params(&params).expect("len 2");
        spec.get_params(&mut params).expect("len 2");
        assert_close(params[1], 3.0_f64.ln());
        let b = spec.parameter_bindings();
        assert_eq!(b.len(), 2);
        assert_eq!(b[0].leaf_id, 0);
        assert_eq!(b[1].leaf_id, 0);
        assert_eq!(b[1].local_index, 1);
        match spec.compile() {
            crate::kernel::CompiledKernel::RbfArd(leaf) => assert_eq!(leaf.num_params(), 2),
            other => panic!("expected ARD RBF, got {other:?}"),
        }
    }

    #[test]
    fn constant_and_white_flatten_with_rbf() {
        let spec = rbf(1.0)
            + KernelSpec::from(ConstantKernel::new(2.0).expect("valid"))
            + KernelSpec::from(WhiteKernel::new(0.1).expect("valid"));
        assert_eq!(spec.num_params(), 3);
        let mut params = [0.0; 3];
        spec.get_params(&mut params).expect("len 3");
        assert_close(params[1], 2.0_f64.ln());
        assert_close(params[2], 0.1_f64.ln());
        let compiled = spec.compile();
        assert_eq!(compiled.num_params(), 3);
        assert_eq!(
            compiled.coord_mode().expect("compat"),
            crate::kernel::CoordMode::Dist
        );
    }

    #[test]
    fn matern_and_matern_ard_flatten() {
        let iso = KernelSpec::from(MaternKernel::new(1.5, MaternNu::ThreeHalves).expect("valid"));
        assert_eq!(iso.num_params(), 1);
        let mut params = [0.0];
        iso.get_params(&mut params).expect("len 1");
        assert_close(params[0], 1.5_f64.ln());
        let mut ard =
            KernelSpec::from(MaternArdKernel::new(&[1.0, 2.0], MaternNu::Half).expect("valid"));
        assert_eq!(ard.num_params(), 2);
        let mut ard_params = [0.0; 2];
        ard.get_params(&mut ard_params).expect("len 2");
        assert_close(ard_params[1], 2.0_f64.ln());
        ard_params[1] = 3.0_f64.ln();
        ard.set_params(&ard_params).expect("len 2");
        match ard.compile() {
            crate::kernel::CompiledKernel::MaternArd(leaf) => {
                assert_eq!(leaf.num_params(), 2);
                assert_eq!(leaf.nu(), MaternNu::Half);
            }
            other => panic!("expected ARD Matern, got {other:?}"),
        }
    }
}
