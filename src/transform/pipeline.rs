//! Sequential composition of input or target maps.

use std::fmt;

use super::{
    IdentityInput, IdentityTarget, TargetTransform, Transform, UnfittedTarget, UnfittedTransform,
};
use crate::error::GprError;

/// Unfitted sequence of input maps. [`Self::fit`] returns [`FittedPipeline`].
///
/// Each later map is fitted on the output of the previous map. An empty
/// pipeline is identity: [`Self::fit`] only checks packing and finiteness.
///
/// Pass this to [`crate::Gpr::with_input_transform`]. A single map still
/// uses that method without a pipeline.
///
/// # Examples
///
/// ```rust
/// use gprx::transform::{MinMaxInput, Pipeline, StandardizeInput, Transform};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let t = Pipeline::new()
///     .then(MinMaxInput::new())
///     .then(StandardizeInput::new())
///     .fit(&[0.0, 2.0, 4.0], 3, 1)?;
/// let mut x = [0.0, 2.0, 4.0];
/// t.apply(&mut x, 3, 1)?;
/// # let _ = x;
/// # Ok(())
/// # }
/// ```
pub struct Pipeline {
    steps: Vec<Box<dyn UnfittedTransform>>,
}

impl Pipeline {
    /// Returns an empty pipeline (identity until a map is appended).
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Appends a map applied after the maps already in this pipeline.
    pub fn then(mut self, step: impl UnfittedTransform + 'static) -> Self {
        self.steps.push(Box::new(step));
        self
    }

    /// Returns the number of maps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns whether this pipeline has no maps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    pub(crate) fn from_steps(steps: Vec<Box<dyn UnfittedTransform>>) -> Self {
        Self { steps }
    }

    pub(crate) fn steps(&self) -> &[Box<dyn UnfittedTransform>] {
        &self.steps
    }

    /// Fits each map on the output of the previous map.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] from the first map that rejects `x`, or
    /// [`GprError::EmptyInput`] / [`GprError::NonFiniteInput`] when this
    /// pipeline is empty and `x` is empty, packed incorrectly, or non-finite.
    pub fn fit(self, x: &[f64], n_rows: usize, n_cols: usize) -> Result<FittedPipeline, GprError> {
        if self.steps.is_empty() {
            IdentityInput.fit(x, n_rows, n_cols)?;
            return Ok(FittedPipeline { steps: Vec::new() });
        }
        let mut buf = x.to_vec();
        let mut fitted = Vec::with_capacity(self.steps.len());
        for step in self.steps {
            let t = step.fit(&buf, n_rows, n_cols)?;
            t.apply(&mut buf, n_rows, n_cols)?;
            fitted.push(t);
        }
        Ok(FittedPipeline { steps: fitted })
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Pipeline {
    fn clone(&self) -> Self {
        Self {
            steps: self.steps.iter().map(|step| step.clone_box()).collect(),
        }
    }
}

impl fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pipeline")
            .field("len", &self.steps.len())
            .finish()
    }
}

impl UnfittedTransform for Pipeline {
    fn fit(
        self: Box<Self>,
        x: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Box<dyn Transform>, GprError> {
        (*self).fit(x, n_rows, n_cols).map(|t| Box::new(t) as _)
    }

    fn clone_box(&self) -> Box<dyn UnfittedTransform> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Fitted sequence of input maps.
///
/// [`Self::apply`] runs each map in the same order as [`Pipeline::then`].
pub struct FittedPipeline {
    steps: Vec<Box<dyn Transform>>,
}

impl FittedPipeline {
    /// Returns the number of maps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns whether this pipeline has no maps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    pub(crate) fn from_steps(steps: Vec<Box<dyn Transform>>) -> Self {
        Self { steps }
    }

    pub(crate) fn steps(&self) -> &[Box<dyn Transform>] {
        &self.steps
    }
}

impl Clone for FittedPipeline {
    fn clone(&self) -> Self {
        Self {
            steps: self.steps.iter().map(|step| step.clone_box()).collect(),
        }
    }
}

impl fmt::Debug for FittedPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FittedPipeline")
            .field("len", &self.steps.len())
            .finish()
    }
}

impl Transform for FittedPipeline {
    fn apply(&self, x: &mut [f64], n_rows: usize, n_cols: usize) -> Result<(), GprError> {
        if self.steps.is_empty() {
            return IdentityInput.apply(x, n_rows, n_cols);
        }
        for step in &self.steps {
            step.apply(x, n_rows, n_cols)?;
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn Transform> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Unfitted sequence of target maps. [`Self::fit`] returns [`FittedTargetPipeline`].
///
/// Each later map is fitted on the output of the previous map. An empty
/// pipeline is identity: [`Self::fit`] only checks that `y` is finite.
///
/// Pass this to [`crate::Gpr::with_target_transform`]. A single map still
/// uses that method without a pipeline.
///
/// # Examples
///
/// ```rust
/// use gprx::transform::{MinMaxTarget, StandardizeTarget, TargetPipeline, TargetTransform};
///
/// # fn main() -> Result<(), gprx::GprError> {
/// let t = TargetPipeline::new()
///     .then(MinMaxTarget::new())
///     .then(StandardizeTarget::new())
///     .fit(&[1.0, 3.0, 5.0])?;
/// let mut y = [1.0, 3.0, 5.0];
/// t.transform(&mut y)?;
/// t.inverse_transform_mean(&mut y)?;
/// # let _ = y;
/// # Ok(())
/// # }
/// ```
pub struct TargetPipeline {
    steps: Vec<Box<dyn UnfittedTarget>>,
}

impl TargetPipeline {
    /// Returns an empty pipeline (identity until a map is appended).
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Appends a map applied after the maps already in this pipeline.
    pub fn then(mut self, step: impl UnfittedTarget + 'static) -> Self {
        self.steps.push(Box::new(step));
        self
    }

    /// Returns the number of maps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns whether this pipeline has no maps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    pub(crate) fn from_steps(steps: Vec<Box<dyn UnfittedTarget>>) -> Self {
        Self { steps }
    }

    pub(crate) fn steps(&self) -> &[Box<dyn UnfittedTarget>] {
        &self.steps
    }

    /// Fits each map on the output of the previous map.
    ///
    /// # Errors
    ///
    /// Returns [`GprError`] from the first map that rejects `y`, or
    /// [`GprError::NonFiniteInput`] when this pipeline is empty and `y`
    /// contains `NaN` or `Inf`.
    pub fn fit(self, y: &[f64]) -> Result<FittedTargetPipeline, GprError> {
        if self.steps.is_empty() {
            IdentityTarget.fit(y)?;
            return Ok(FittedTargetPipeline { steps: Vec::new() });
        }
        let mut buf = y.to_vec();
        let mut fitted = Vec::with_capacity(self.steps.len());
        for step in self.steps {
            let t = step.fit(&buf)?;
            t.transform(&mut buf)?;
            fitted.push(t);
        }
        Ok(FittedTargetPipeline { steps: fitted })
    }
}

impl Default for TargetPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for TargetPipeline {
    fn clone(&self) -> Self {
        Self {
            steps: self.steps.iter().map(|step| step.clone_box()).collect(),
        }
    }
}

impl fmt::Debug for TargetPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TargetPipeline")
            .field("len", &self.steps.len())
            .finish()
    }
}

impl UnfittedTarget for TargetPipeline {
    fn fit(self: Box<Self>, y: &[f64]) -> Result<Box<dyn TargetTransform>, GprError> {
        (*self).fit(y).map(|t| Box::new(t) as _)
    }

    fn clone_box(&self) -> Box<dyn UnfittedTarget> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Fitted sequence of target maps.
///
/// Forward maps run in [`TargetPipeline::then`] order. Inverse mean and
/// variance run in reverse, so a MinMax-then-Standardize stack undoes
/// Standardize first.
pub struct FittedTargetPipeline {
    steps: Vec<Box<dyn TargetTransform>>,
}

impl FittedTargetPipeline {
    /// Returns the number of maps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns whether this pipeline has no maps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    pub(crate) fn from_steps(steps: Vec<Box<dyn TargetTransform>>) -> Self {
        Self { steps }
    }

    pub(crate) fn steps(&self) -> &[Box<dyn TargetTransform>] {
        &self.steps
    }
}

impl Clone for FittedTargetPipeline {
    fn clone(&self) -> Self {
        Self {
            steps: self.steps.iter().map(|step| step.clone_box()).collect(),
        }
    }
}

impl fmt::Debug for FittedTargetPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FittedTargetPipeline")
            .field("len", &self.steps.len())
            .finish()
    }
}

impl TargetTransform for FittedTargetPipeline {
    fn transform(&self, y: &mut [f64]) -> Result<(), GprError> {
        if self.steps.is_empty() {
            return IdentityTarget.transform(y);
        }
        for step in &self.steps {
            step.transform(y)?;
        }
        Ok(())
    }

    fn inverse_transform_mean(&self, mean: &mut [f64]) -> Result<(), GprError> {
        if self.steps.is_empty() {
            return IdentityTarget.inverse_transform_mean(mean);
        }
        for step in self.steps.iter().rev() {
            step.inverse_transform_mean(mean)?;
        }
        Ok(())
    }

    fn inverse_transform_variance(&self, var: &mut [f64]) -> Result<(), GprError> {
        if self.steps.is_empty() {
            return IdentityTarget.inverse_transform_variance(var);
        }
        for step in self.steps.iter().rev() {
            step.inverse_transform_variance(var)?;
        }
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn TargetTransform> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FittedPipeline, FittedTargetPipeline, Pipeline, TargetPipeline, TargetTransform, Transform,
    };
    use crate::error::GprError;
    use crate::transform::{MinMaxInput, MinMaxTarget, StandardizeInput, StandardizeTarget};

    const TOL: f64 = 1e-10;

    use crate::test_check::{assert_close, assert_send_sync};

    #[test]
    fn is_send_sync() {
        assert_send_sync::<Pipeline>();
        assert_send_sync::<FittedPipeline>();
        assert_send_sync::<TargetPipeline>();
        assert_send_sync::<FittedTargetPipeline>();
    }

    #[test]
    fn empty_input_pipeline_is_identity() {
        let t = Pipeline::new()
            .fit(&[1.0, 2.0, 3.0, 4.0], 2, 2)
            .expect("valid");
        assert!(t.is_empty());
        let mut x = [1.0, 2.0, 3.0, 4.0];
        t.apply(&mut x, 2, 2).expect("valid");
        assert_close(x[0], 1.0, TOL);
        assert_close(x[3], 4.0, TOL);
    }

    #[test]
    fn input_minmax_then_standardize_matches_manual() {
        let x = [0.0, 2.0, 4.0, 1.0, 3.0, 5.0];
        let pipeline = Pipeline::new()
            .then(MinMaxInput::new())
            .then(StandardizeInput::new());
        assert_eq!(pipeline.len(), 2);
        let t = pipeline.fit(&x, 3, 2).expect("valid");
        let minmax = MinMaxInput::new().fit(&x, 3, 2).expect("valid");
        let mut mid = x;
        minmax.apply(&mut mid, 3, 2).expect("ok");
        let std = StandardizeInput::new().fit(&mid, 3, 2).expect("valid");
        let mut expected = mid;
        std.apply(&mut expected, 3, 2).expect("ok");
        let mut got = x;
        t.apply(&mut got, 3, 2).expect("ok");
        for (actual, want) in got.iter().zip(expected.iter()) {
            assert_close(*actual, *want, TOL);
        }
    }

    #[test]
    fn empty_input_pipeline_rejects() {
        assert!(matches!(
            Pipeline::new().fit(&[], 0, 1),
            Err(GprError::EmptyInput)
        ));
        assert!(matches!(
            Pipeline::new().fit(&[1.0, f64::NAN], 1, 2),
            Err(GprError::NonFiniteInput)
        ));
    }

    #[test]
    fn empty_target_pipeline_is_identity() {
        let t = TargetPipeline::new().fit(&[1.0, 2.0]).expect("finite");
        assert!(t.is_empty());
        let mut y = [1.0, 2.0];
        t.transform(&mut y).expect("finite");
        t.inverse_transform_mean(&mut y).expect("finite");
        t.inverse_transform_variance(&mut y).expect("finite");
        assert_close(y[0], 1.0, TOL);
        assert_close(y[1], 2.0, TOL);
    }

    #[test]
    fn target_minmax_then_standardize_roundtrips() {
        let y = [1.0, 3.0, 5.0];
        let pipeline = TargetPipeline::new()
            .then(MinMaxTarget::new())
            .then(StandardizeTarget::new());
        assert_eq!(pipeline.len(), 2);
        let t = pipeline.fit(&y).expect("valid");
        let minmax = MinMaxTarget::new().fit(&y).expect("valid");
        let mut mid = y;
        minmax.transform(&mut mid).expect("ok");
        let std = StandardizeTarget::new().fit(&mid).expect("valid");
        let mut expected = mid;
        std.transform(&mut expected).expect("ok");
        let mut got = y;
        t.transform(&mut got).expect("ok");
        for (actual, want) in got.iter().zip(expected.iter()) {
            assert_close(*actual, *want, TOL);
        }
        t.inverse_transform_mean(&mut got).expect("ok");
        assert_close(got[0], y[0], TOL);
        assert_close(got[1], y[1], TOL);
        assert_close(got[2], y[2], TOL);
        let s = std.std();
        let minmax_scale = (minmax.max() - minmax.min()) / (1.0 - 0.0);
        let mut var = [1.0, 0.25];
        t.inverse_transform_variance(&mut var).expect("ok");
        let scale2 = (s * minmax_scale).powi(2);
        assert_close(var[0], 1.0 * scale2, TOL);
        assert_close(var[1], 0.25 * scale2, TOL);
    }

    #[test]
    fn empty_target_pipeline_rejects_non_finite() {
        assert!(matches!(
            TargetPipeline::new().fit(&[f64::INFINITY]),
            Err(GprError::NonFiniteInput)
        ));
    }

    #[test]
    fn clone_keeps_step_count() {
        let p = Pipeline::new()
            .then(MinMaxInput::new())
            .then(StandardizeInput::new());
        assert_eq!(p.clone().len(), 2);
        let q = TargetPipeline::new()
            .then(MinMaxTarget::new())
            .then(StandardizeTarget::new());
        assert_eq!(q.clone().len(), 2);
    }
}
