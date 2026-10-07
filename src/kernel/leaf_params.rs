//! The parameter vector of one leaf: its length, its values and intervals
//! written at an offset, and new values read from one. A [`super::KernelSpec`],
//! a [`super::CompiledKernel`], and a supplied-distance leaf share these.

use crate::error::GprError;
use crate::kernel::{
    ConstantKernel, CustomKernel, KernelScalar, LinearKernel, MaternArdKernel, MaternKernel,
    PeriodicKernel, RationalQuadraticArdKernel, RationalQuadraticKernel, RbfArdKernel, RbfKernel,
    WhiteKernel,
};
use crate::param::Interval;

/// One leaf's slice of the flattened `θ`.
pub(crate) trait LeafParams {
    /// Number of `θ` entries the leaf owns.
    fn leaf_num_params(&self) -> usize;

    /// Writes the leaf's `θ` at `*offset` and advances it.
    fn write_leaf_params(&self, out: &mut [f64], offset: &mut usize) -> Result<(), GprError>;

    /// Writes the leaf's open intervals at `*offset` and advances it.
    fn write_leaf_intervals(
        &self,
        out: &mut [Interval],
        offset: &mut usize,
    ) -> Result<(), GprError>;

    /// Reads the leaf's `θ` from `params` at `*offset` and advances it.
    fn apply_leaf_params(&mut self, params: &[f64], offset: &mut usize) -> Result<(), GprError>;
}

/// `set_params` on the leaf's slice of `params`.
macro_rules! apply_slice {
    () => {
        fn apply_leaf_params(
            &mut self,
            params: &[f64],
            offset: &mut usize,
        ) -> Result<(), GprError> {
            let n = self.num_params();
            self.set_params(&params[*offset..*offset + n])?;
            *offset += n;
            Ok(())
        }
    };
}

/// A leaf with one positive parameter `log(v)` and its interval.
macro_rules! one_param {
    ($leaf:ty, $log:ident) => {
        impl LeafParams for $leaf {
            fn leaf_num_params(&self) -> usize {
                1
            }

            fn write_leaf_params(
                &self,
                out: &mut [f64],
                offset: &mut usize,
            ) -> Result<(), GprError> {
                out[*offset] = self.$log();
                *offset += 1;
                Ok(())
            }

            fn write_leaf_intervals(
                &self,
                out: &mut [Interval],
                offset: &mut usize,
            ) -> Result<(), GprError> {
                out[*offset] = self.bounds();
                *offset += 1;
                Ok(())
            }

            apply_slice!();
        }
    };
}

one_param!(RbfKernel, log_lengthscale);
one_param!(MaternKernel, log_lengthscale);
one_param!(ConstantKernel, log_constant);
one_param!(LinearKernel, log_variance);
one_param!(WhiteKernel, log_variance);

/// An ARD leaf without other parameters: one `log(ℓ_d)` per dimension.
macro_rules! ard_only {
    ($leaf:ty) => {
        impl LeafParams for $leaf {
            fn leaf_num_params(&self) -> usize {
                self.num_params()
            }

            fn write_leaf_params(
                &self,
                out: &mut [f64],
                offset: &mut usize,
            ) -> Result<(), GprError> {
                let n = self.num_params();
                out[*offset..*offset + n].copy_from_slice(self.log_lengthscales());
                *offset += n;
                Ok(())
            }

            fn write_leaf_intervals(
                &self,
                out: &mut [Interval],
                offset: &mut usize,
            ) -> Result<(), GprError> {
                self.lengthscales().write_intervals(out, offset);
                Ok(())
            }

            apply_slice!();
        }
    };
}

ard_only!(RbfArdKernel);
ard_only!(MaternArdKernel);

impl LeafParams for PeriodicKernel {
    fn leaf_num_params(&self) -> usize {
        2
    }

    fn write_leaf_params(&self, out: &mut [f64], offset: &mut usize) -> Result<(), GprError> {
        out[*offset] = self.log_lengthscale();
        out[*offset + 1] = self.log_period();
        *offset += 2;
        Ok(())
    }

    fn write_leaf_intervals(
        &self,
        out: &mut [Interval],
        offset: &mut usize,
    ) -> Result<(), GprError> {
        out[*offset] = self.lengthscale_bounds();
        out[*offset + 1] = self.period_bounds();
        *offset += 2;
        Ok(())
    }

    apply_slice!();
}

impl LeafParams for RationalQuadraticKernel {
    fn leaf_num_params(&self) -> usize {
        2
    }

    fn write_leaf_params(&self, out: &mut [f64], offset: &mut usize) -> Result<(), GprError> {
        out[*offset] = self.log_lengthscale();
        out[*offset + 1] = self.log_alpha();
        *offset += 2;
        Ok(())
    }

    fn write_leaf_intervals(
        &self,
        out: &mut [Interval],
        offset: &mut usize,
    ) -> Result<(), GprError> {
        out[*offset] = self.lengthscale_bounds();
        out[*offset + 1] = self.alpha_bounds();
        *offset += 2;
        Ok(())
    }

    apply_slice!();
}

impl LeafParams for RationalQuadraticArdKernel {
    fn leaf_num_params(&self) -> usize {
        self.num_params()
    }

    fn write_leaf_params(&self, out: &mut [f64], offset: &mut usize) -> Result<(), GprError> {
        let n = self.lengthscales().num_params();
        out[*offset..*offset + n].copy_from_slice(self.log_lengthscales());
        out[*offset + n] = self.log_alpha();
        *offset += n + 1;
        Ok(())
    }

    fn write_leaf_intervals(
        &self,
        out: &mut [Interval],
        offset: &mut usize,
    ) -> Result<(), GprError> {
        self.lengthscales().write_intervals(out, offset);
        out[*offset] = self.alpha_bounds();
        *offset += 1;
        Ok(())
    }

    apply_slice!();
}

impl<T: KernelScalar> LeafParams for CustomKernel<T> {
    fn leaf_num_params(&self) -> usize {
        self.num_params()
    }

    fn write_leaf_params(&self, out: &mut [f64], offset: &mut usize) -> Result<(), GprError> {
        self.write_params(out, offset)
    }

    fn write_leaf_intervals(
        &self,
        out: &mut [Interval],
        offset: &mut usize,
    ) -> Result<(), GprError> {
        self.write_intervals(out, offset)
    }

    fn apply_leaf_params(&mut self, params: &[f64], offset: &mut usize) -> Result<(), GprError> {
        self.apply_params(params, offset)
    }
}
