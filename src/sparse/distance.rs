//! The supplied distances of a sparse prediction, and the predict family of
//! a sparse model of a [`DistanceKernel`](crate::kernel::DistanceKernel).

use crate::error::GprError;
use crate::kernel::{BlockKind, DistanceSource, QuerySources, spec_slots};

use super::{QueryDist, SparseCore};

impl QueryDist {
    /// Binds the train × query blocks (`n × q`) and, for a covariance, the
    /// query × query squares (`q × q`) of `core`, and keeps their inducing
    /// rows.
    ///
    /// # Errors
    ///
    /// Returns [`GprError::EmptyInput`] if `q` is zero, and the errors of
    /// binding the sources.
    pub(crate) fn bind<'s>(
        core: &SparseCore,
        cross: Vec<DistanceSource<'s>>,
        square: Option<Vec<DistanceSource<'s>>>,
        q: usize,
    ) -> Result<Self, GprError> {
        crate::data::require_nonempty(q)?;
        let inducing = core
            .dist
            .as_ref()
            .map_or(&[][..], |dist| &dist.inducing[..]);
        let slots = spec_slots(&core.kernel);
        let zq = QuerySources::<f64>::bind(&slots, cross, core.n, q, BlockKind::Rect)?
            .gather_rows(inducing);
        let qq = square
            .map(|square| {
                let all: Vec<usize> = (0..q).collect();
                QuerySources::<f64>::bind(&slots, square, q, q, BlockKind::Square)
                    .map(|square| square.gather_rows(&all))
            })
            .transpose()?;
        Ok(Self { zq: Some(zq), qq })
    }
}

/// The [`DistanceQuery`](crate::prediction::DistanceQuery) of a sparse
/// model: `impl [generics] Model`. The model keeps the inducing rows of the
/// caller's blocks for one call and runs its coordinate query on them.
macro_rules! sparse_query {
    (impl [$($gen:tt)*] $model:ty) => {
        impl<$($gen)*> $crate::prediction::DistanceQuery for $model {
            type Refine = P::Refine;

            fn query_distances(
                &self,
                cross: Vec<DistanceSource<'_>>,
                points: $crate::prediction::QueryPoints<'_>,
                q: usize,
                options: PredictOptions,
            ) -> Result<Prediction<P::Refine>, GprError> {
                let qd = QueryDist::bind(&self.core, cross, None, q)?;
                self.query(points.xs, q, points.n_cols, &qd, options)
            }

            fn query_distances_into(
                &mut self,
                cross: Vec<DistanceSource<'_>>,
                points: $crate::prediction::QueryPoints<'_>,
                q: usize,
                options: PredictOptions,
                out: &mut Prediction<P::Refine>,
            ) -> Result<(), GprError> {
                let qd = QueryDist::bind(&self.core, cross, None, q)?;
                self.query_into(points.xs, q, points.n_cols, &qd, options, out)
            }

            fn query_distance_covariance(
                &self,
                cross: Vec<DistanceSource<'_>>,
                square: Vec<DistanceSource<'_>>,
                points: $crate::prediction::QueryPoints<'_>,
                q: usize,
                options: PredictOptions,
            ) -> Result<PredictiveCovariance<P::Refine>, GprError> {
                let qd = QueryDist::bind(&self.core, cross, Some(square), q)?;
                self.query_covariance(points.xs, q, points.n_cols, &qd, options)
            }

            fn draw_jitter(&self) -> $crate::policy::JitterPolicy {
                self.core.jitter
            }
        }
    };
}

pub(crate) use sparse_query;

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use crate::kernel::{KernelSpec, RbfKernel, ScalarDistance};
    use crate::sparse::{SparseDist, cast_blocks, zx_at, zz_at};

    fn dist() -> SparseDist {
        let image = ScalarDistance::new();
        let kernel: KernelSpec = image
            .kernel(RbfKernel::new(1.0).expect("ell"))
            .spec()
            .clone();
        let d2 = vec![0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0];
        SparseDist::bind(&kernel, vec![image.from_vec(d2)], 3, &[0, 2]).expect("bind")
    }

    #[test]
    fn an_f64_model_borrows_the_blocks() {
        let dist = dist();
        let zz = zz_at::<f64>(Some(&dist)).expect("zz");
        assert!(matches!(zz, Some(Cow::Borrowed(_))));
        let zx = zx_at::<f64>(Some(&dist)).expect("zx");
        assert!(matches!(zx, Some(Cow::Borrowed(_))));
        assert!(matches!(
            cast_blocks::<f64>(Some(&dist.zx)),
            Some(Cow::Borrowed(_))
        ));
    }

    #[test]
    fn an_f32_model_casts_the_blocks_once() {
        let dist = dist();
        let first = zx_at::<f32>(Some(&dist)).expect("zx");
        let again = zx_at::<f32>(Some(&dist)).expect("zx");
        let (Some(Cow::Borrowed(first)), Some(Cow::Borrowed(again))) = (first, again) else {
            panic!("borrowed");
        };
        assert!(std::ptr::eq(first, again));
        let zz = zz_at::<f32>(Some(&dist)).expect("zz");
        assert!(matches!(zz, Some(Cow::Borrowed(_))));
    }
}
