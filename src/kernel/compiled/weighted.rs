//! Every `⟨V, ∂K/∂θ⟩` of a kernel tree in one walk of the tree.
//!
//! The joint gradient of a fit needs `⟨W, ∂K/∂θ_i⟩_F` for every `θ_i`.
//! Writing `∂K/∂θ_i` one parameter at a time evaluates the other factors of
//! a product again for each parameter. This walk passes a weight matrix down
//! the tree instead: a sum hands its weight to every term unchanged, and a
//! product hands factor `c` the weight `V ∘ ∏_{s≠c} K_s`, since
//! `⟨V, ∏_{s≠c} K_s ∘ ∂K_c⟩ = ⟨V ∘ ∏_{s≠c} K_s, ∂K_c⟩`.
//!
//! A constant factor is a scalar: it scales the weight and needs no matrix.
//! Its own derivative is `⟨V, K_product⟩`, which the walk returns from the
//! pass of another factor. A product with one non-constant factor therefore
//! evaluates no Gram at all. A product with two or more reads the factor
//! Grams the factor step kept ([`CompiledKernel::eval_gram_keeping`]) when
//! the budget allowed it, and evaluates them otherwise. A leaf that is handed
//! its own Gram builds `∂K/∂θ` from those values instead of the
//! transcendental functions. Every matrix of the square walk is the lower
//! triangle of a symmetric `n × n`. The rectangular walk is a full matrix.
//! An ARD RBF rectangle contracts every lengthscale from one `exp` and does
//! not form `∂K`. An Accurate square ARD RBF forms `k` once and folds every
//! lengthscale from that `k` as one matrix product. The diagonal walk sums
//! `∂k(x_i, x_i)/∂θ`.

use super::gram::GramInputs;
use super::supplied::{ArdLeaf, ArdSquare, ScalarLeaf, SuppliedCompiled, SuppliedLeaf};
use super::{CompiledKernel, CrossViews, add_triangle};
use crate::error::GprError;
use crate::kernel::dist::{for_each_lower_col, lower_fold_infallible};
use crate::kernel::tree::{NoSupply, Supply};
use crate::kernel::{KernelScalar, Triangle};
use faer::{Mat, MatMut, MatRef};
use std::ops::Range;

/// Where the walk is: the inputs, the scratch it shares, and the kept
/// factor Grams with the number of products allowed to read them.
pub(crate) struct WeightedWalk<'a, 'b, T: KernelScalar, S: Supply = NoSupply> {
    pub(crate) inputs: GramInputs<'a, T, S>,
    pub(crate) scratch: MatMut<'b, T>,
    pub(crate) nested: &'b mut Vec<Mat<T>>,
    /// The Grams [`CompiledKernel::eval_gram_keeping`] wrote, in walk order.
    pub(crate) kept: &'b [Mat<T>],
    /// How many products read `kept` ([`CompiledKernel::kept_products`]).
    pub(crate) kept_products: usize,
    /// Workspace for the ARD lengthscale matrix product. The caller keeps it
    /// across evaluations.
    pub(crate) fold: &'b mut Vec<f64>,
}

impl<T: KernelScalar, S: Supply> CompiledKernel<T, S> {
    /// Factors of this product that are not [`Self::Constant`].
    fn varying_factors(terms: &[Self]) -> usize {
        terms
            .iter()
            .filter(|t| !matches!(t, Self::Constant(_)))
            .count()
    }

    /// `∏ c` over the constant factors of a product.
    fn constant_scale(terms: &[Self]) -> f64 {
        terms
            .iter()
            .map(|t| match t {
                Self::Constant(leaf) => leaf.constant(),
                _ => 1.0,
            })
            .product()
    }

    /// The root terms (the terms of a root sum, else the root itself) in
    /// walk order, each with the range of the kept Grams its factor Grams
    /// take when it is one of the first `products` products that keep them.
    ///
    /// A product can keep its Grams when it has two or more non-constant
    /// factors. Sums are flattened when the tree is compiled, so no other
    /// product is reached through sums only. This is the one place that
    /// decides which products keep their Grams and where those Grams are;
    /// the buffer count, [`Self::eval_gram_keeping`], and
    /// [`Self::weighted_grads`] all read it.
    fn keep_plan(&self, products: usize) -> impl Iterator<Item = (&Self, Option<Range<usize>>)> {
        let roots: &[Self] = match self {
            Self::Sum(terms) => terms,
            other => std::slice::from_ref(other),
        };
        let (mut kept, mut offset) = (0, 0);
        roots.iter().map(move |t| {
            let slot = match t {
                Self::Product(factors)
                    if Self::varying_factors(factors) >= 2 && kept < products =>
                {
                    kept += 1;
                    let grams = offset..offset + Self::varying_factors(factors);
                    offset = grams.end;
                    Some(grams)
                }
                _ => None,
            };
            (t, slot)
        })
    }

    /// `n×n` Grams kept for the first `products` products.
    fn kept_grams(&self, products: usize) -> usize {
        self.keep_plan(products)
            .filter_map(|(_, slot)| slot)
            .map(|grams| grams.len())
            .sum()
    }

    /// How many products keep their factor Grams: the most for which the
    /// kept Grams and the walk's own buffers stay within
    /// [`Self::unkept_buffers`], the buffers a walk that keeps nothing and
    /// evaluates every factor (constants included) would take.
    pub(crate) fn kept_products(&self) -> usize {
        let budget = self.unkept_buffers();
        let candidates = self
            .keep_plan(usize::MAX)
            .filter(|(_, slot)| slot.is_some())
            .count();
        (0..=candidates)
            .rev()
            .find(|&p| self.kept_grams(p) + self.walk_buffers(p) <= budget)
            .unwrap_or(0)
    }

    /// The `n × n` buffers of the joint gradient when the first `products`
    /// products keep their Grams: the kept Grams first, then the walk's own.
    pub(crate) fn weighted_buffers(&self, products: usize) -> usize {
        self.kept_grams(products) + self.walk_buffers(products)
    }

    /// The kept Grams [`Self::weighted_buffers`] starts with.
    pub(crate) fn kept_buffers(&self, products: usize) -> usize {
        self.kept_grams(products)
    }

    /// Buffers of a walk that evaluates every product factor: each factor's
    /// Gram, the weight handed down, and the deepest term.
    fn unkept_buffers(&self) -> usize {
        match self {
            Self::Sum(terms) => terms.iter().map(Self::unkept_buffers).max().unwrap_or(0),
            Self::Product(terms) => {
                terms.len() + 1 + terms.iter().map(Self::unkept_buffers).max().unwrap_or(0)
            }
            _ => 1,
        }
    }

    /// Buffers of one contraction that keeps no Grams ([`Self::weighted_grads`]
    /// with `kept_products = 0`, and the rectangular and diagonal walks).
    pub(crate) fn contraction_buffers(&self) -> usize {
        self.walk_buffers(0)
    }

    /// The walk's own buffers when the first `products` products read kept Grams.
    fn walk_buffers(&self, products: usize) -> usize {
        self.keep_plan(products)
            .map(|(t, slot)| t.node_buffers(slot.is_some()))
            .max()
            .unwrap_or(0)
    }

    /// Buffers of one node below the root. `keeps` says a product reads kept Grams.
    fn node_buffers(&self, keeps: bool) -> usize {
        match self {
            Self::Sum(terms) => terms
                .iter()
                .map(|t| t.node_buffers(false))
                .max()
                .unwrap_or(0),
            Self::Product(terms) => {
                let varying = Self::varying_factors(terms);
                let deepest = terms
                    .iter()
                    .filter(|t| !matches!(t, Self::Constant(_)))
                    .map(|t| t.node_buffers(false))
                    .max()
                    .unwrap_or(0);
                match varying {
                    0 => 0,
                    1 => deepest,
                    _ => usize::from(!keeps) * varying + 1 + deepest,
                }
            }
            Self::Constant(_) => 0,
            _ => 1,
        }
    }

    /// Writes `K` for the lower triangle like [`Self::eval_gram`], keeping
    /// the factor Grams of the first `products` candidate products in
    /// `kept` (walk order) for [`Self::weighted_grads`].
    ///
    /// `scratch` and `term` are two distinct `out`-shaped buffers.
    // The inputs, the output, two scratch buffers, the levels, and the kept Grams.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn eval_gram_keeping<M: crate::math::KernelMath>(
        &self,
        inputs: GramInputs<'_, T, S>,
        mut out: MatMut<'_, T>,
        mut scratch: MatMut<'_, T>,
        mut term: MatMut<'_, T>,
        nested: &mut Vec<Mat<T>>,
        kept: &mut [Mat<T>],
        products: usize,
    ) -> Result<(), GprError> {
        self.require_tree_columns(inputs.x)?;
        if products == 0 {
            return self.eval_gram::<M>(inputs, out, Triangle::Lower, scratch, nested);
        }
        let root_is_sum = matches!(self, Self::Sum(_));
        for (i, (t, slot)) in self.keep_plan(products).enumerate() {
            match (t, slot) {
                (Self::Product(factors), Some(range)) => {
                    let grams = &mut kept[range];
                    Self::keep_factor_grams::<M>(factors, inputs, grams, scratch.as_mut(), nested)?;
                    write_scaled_product(out.as_mut(), Self::constant_scale(factors), grams, i > 0);
                }
                // The root itself, with nothing kept: straight into `out`.
                _ if !root_is_sum => {
                    t.eval_gram::<M>(
                        inputs,
                        out.as_mut(),
                        Triangle::Lower,
                        scratch.as_mut(),
                        nested,
                    )?;
                }
                _ => {
                    t.eval_gram::<M>(
                        inputs,
                        term.as_mut(),
                        Triangle::Lower,
                        scratch.as_mut(),
                        nested,
                    )?;
                    if i == 0 {
                        copy_lower(out.as_mut(), term.as_ref());
                    } else {
                        add_triangle(out.as_mut(), term.as_ref(), Triangle::Lower);
                    }
                }
            }
        }
        Ok(())
    }

    /// Evaluates every non-constant factor into its kept Gram.
    fn keep_factor_grams<M: crate::math::KernelMath>(
        factors: &[Self],
        inputs: GramInputs<'_, T, S>,
        grams: &mut [Mat<T>],
        mut scratch: MatMut<'_, T>,
        nested: &mut Vec<Mat<T>>,
    ) -> Result<(), GprError> {
        let varying = factors.iter().filter(|t| !matches!(t, Self::Constant(_)));
        for (factor, gram) in varying.zip(grams.iter_mut()) {
            factor.eval_gram::<M>(
                inputs,
                gram.as_mut(),
                Triangle::Lower,
                scratch.as_mut(),
                nested,
            )?;
        }
        Ok(())
    }

    /// Writes `⟨weight, ∂K/∂θ_p⟩_F` for every parameter `p` of this tree
    /// into `out` (length [`Self::num_params`]), from the lower triangles.
    ///
    /// `bufs` holds the walk's own buffers: [`Self::weighted_buffers`] minus
    /// [`Self::kept_buffers`] matrices of `weight`'s shape. Their contents
    /// are overwritten.
    pub(crate) fn weighted_grads<M: crate::math::KernelMath>(
        &self,
        walk: &mut WeightedWalk<'_, '_, T, S>,
        weight: MatRef<'_, T>,
        out: &mut [f64],
        bufs: &mut [Mat<T>],
    ) -> Result<(), GprError> {
        self.require_tree_columns(walk.inputs.x)?;
        let kept = walk.kept;
        let mut offset = 0;
        for (t, slot) in self.keep_plan(walk.kept_products) {
            let count = t.num_params();
            t.walk::<M>(
                walk,
                weight,
                &mut out[offset..offset + count],
                bufs,
                Node {
                    grams: slot.map(|range| &kept[range]),
                    own: None,
                    value: false,
                },
            )?;
            offset += count;
        }
        Ok(())
    }

    /// One node of [`Self::weighted_grads`]. Returns `⟨weight, K⟩_F` when
    /// `node.value` asks for it (else whatever the pass found on the way).
    fn walk<M: crate::math::KernelMath>(
        &self,
        walk: &mut WeightedWalk<'_, '_, T, S>,
        weight: MatRef<'_, T>,
        out: &mut [f64],
        bufs: &mut [Mat<T>],
        node: Node<'_, T>,
    ) -> Result<f64, GprError> {
        match self {
            Self::Sum(terms) => {
                let mut offset = 0;
                let mut value = 0.0;
                for t in terms {
                    let count = t.num_params();
                    value += t.walk::<M>(
                        walk,
                        weight,
                        &mut out[offset..offset + count],
                        bufs,
                        Node {
                            grams: None,
                            own: None,
                            value: node.value,
                        },
                    )?;
                    offset += count;
                }
                Ok(value)
            }
            Self::Product(terms) => self.walk_product::<M>(terms, walk, weight, out, bufs, node),
            Self::Constant(leaf) => {
                let value = leaf.constant() * lower_sum(weight);
                out[0] = value;
                Ok(value)
            }
            _ => self.walk_leaf::<M>(walk, weight, out, bufs, node),
        }
    }

    fn walk_product<M: crate::math::KernelMath>(
        &self,
        terms: &[Self],
        walk: &mut WeightedWalk<'_, '_, T, S>,
        weight: MatRef<'_, T>,
        out: &mut [f64],
        bufs: &mut [Mat<T>],
        node: Node<'_, T>,
    ) -> Result<f64, GprError> {
        let scale = Self::constant_scale(terms);
        let has_constant = terms.iter().any(|t| matches!(t, Self::Constant(_)));
        let need_value = node.value || has_constant;
        let varying = Self::varying_factors(terms);
        // `⟨V, K_product⟩`, the derivative of every constant factor.
        let value = match varying {
            0 => scale * lower_sum(weight),
            1 => {
                let mut offset = 0;
                let mut value = 0.0;
                for t in terms {
                    let count = t.num_params();
                    if !matches!(t, Self::Constant(_)) {
                        let slot = &mut out[offset..offset + count];
                        let inner = t.walk::<M>(
                            walk,
                            weight,
                            slot,
                            bufs,
                            Node {
                                grams: None,
                                own: None,
                                value: need_value,
                            },
                        )?;
                        for g in slot.iter_mut() {
                            *g *= scale;
                        }
                        value = scale * inner;
                    }
                    offset += count;
                }
                value
            }
            _ => {
                let (evaluated, rest) = match node.grams {
                    Some(_) => bufs.split_at_mut(0),
                    None => {
                        if bufs.len() < varying {
                            return Err(too_few_buffers());
                        }
                        bufs.split_at_mut(varying)
                    }
                };
                let Some((handed, deeper)) = rest.split_first_mut() else {
                    return Err(too_few_buffers());
                };
                if node.grams.is_none() {
                    let factors = terms.iter().filter(|t| !matches!(t, Self::Constant(_)));
                    for (factor, gram) in factors.zip(evaluated.iter_mut()) {
                        factor.eval_gram::<M>(
                            walk.inputs,
                            gram.as_mut(),
                            Triangle::Lower,
                            walk.scratch.as_mut(),
                            walk.nested,
                        )?;
                    }
                }
                let grams: &[Mat<T>] = match node.grams {
                    Some(kept) => kept,
                    None => evaluated,
                };
                let mut offset = 0;
                let mut c = 0;
                let mut value = 0.0;
                for t in terms {
                    let count = t.num_params();
                    if matches!(t, Self::Constant(_)) {
                        offset += count;
                        continue;
                    }
                    write_handed(handed.as_mut(), weight, scale, grams, c);
                    // A leaf finds `⟨handed, K_c⟩ = ⟨V, K_product⟩` in its own
                    // pass; a nested sum or product reads it from its Gram.
                    let leaf = !matches!(t, Self::Sum(_) | Self::Product(_));
                    let inner = t.walk::<M>(
                        walk,
                        handed.as_ref(),
                        &mut out[offset..offset + count],
                        deeper,
                        Node {
                            grams: None,
                            own: Some(grams[c].as_ref()),
                            value: need_value && c == 0 && leaf,
                        },
                    )?;
                    if need_value && c == 0 {
                        value = if leaf {
                            inner
                        } else {
                            lower_dot(handed.as_ref(), grams[c].as_ref())
                        };
                    }
                    c += 1;
                    offset += count;
                }
                value
            }
        };
        let mut offset = 0;
        for t in terms {
            if matches!(t, Self::Constant(_)) {
                out[offset] = value;
            }
            offset += t.num_params();
        }
        Ok(value)
    }

    fn walk_leaf<M: crate::math::KernelMath>(
        &self,
        walk: &mut WeightedWalk<'_, '_, T, S>,
        weight: MatRef<'_, T>,
        out: &mut [f64],
        bufs: &mut [Mat<T>],
        node: Node<'_, T>,
    ) -> Result<f64, GprError> {
        // Leaves whose parameters share each entry's transcendental work:
        // one pass, no `∂K` matrix. A scalar supplied leaf takes the same
        // paths on its slot's distances; anything else takes the `∂K` path
        // below, which reads the slot itself.
        let (leaf, dist) = match (self, walk.inputs.dist) {
            (Self::Supplied(supplied), _) => match S::square_leaf(supplied, walk.inputs.slots) {
                (
                    SuppliedLeaf {
                        at,
                        leaf: SuppliedCompiled::Scalar(leaf),
                        ..
                    },
                    slots,
                ) => (FastLeaf::from_scalar(leaf), Some(slots.scalar(*at))),
                (SuppliedLeaf { .. }, _) => (FastLeaf::None, None),
            },
            (Self::Periodic(leaf), dist) => (FastLeaf::Periodic(leaf), dist),
            (Self::RationalQuadratic(leaf), dist) => (FastLeaf::RationalQuadratic(leaf), dist),
            (Self::Rbf(leaf), dist) => (FastLeaf::Rbf(leaf), dist),
            _ => (FastLeaf::None, None),
        };
        if let Some(dist) = dist {
            match leaf {
                FastLeaf::Periodic(leaf) => {
                    return leaf.weighted_grads_dist::<M, T>(dist, node.own, weight, out);
                }
                FastLeaf::RationalQuadratic(leaf) => {
                    return leaf.weighted_grads_dist(dist, node.own, weight, out);
                }
                // From `k` when the leaf has it or must form it for `⟨V, K⟩`
                // anyway; otherwise the `∂K` path below, one `exp` per entry too.
                FastLeaf::Rbf(leaf) if M::ACCURATE && (node.own.is_some() || node.value) => {
                    let k = match node.own {
                        Some(k) => k,
                        None => {
                            let Some(gram) = bufs.first_mut() else {
                                return Err(too_few_buffers());
                            };
                            leaf.apply_math::<M, _>(dist, gram.as_mut(), Triangle::Lower)?;
                            gram.as_ref()
                        }
                    };
                    return leaf.weighted_grads_from_gram(dist, k, weight, out);
                }
                FastLeaf::Rbf(_) | FastLeaf::None => {}
            }
        }
        // An accurate ARD RBF on a supplied slot's packed `(Δ_d)²` (the
        // training store): the same one Gram, then every lengthscale from
        // the triangles. Dense blocks take the `∂K` path below.
        if M::ACCURATE
            && let Self::Supplied(supplied) = self
            && let (
                SuppliedLeaf {
                    at,
                    leaf: SuppliedCompiled::Ard(ArdLeaf::Rbf(rbf)),
                    ..
                },
                slots,
            ) = S::square_leaf(supplied, walk.inputs.slots)
            && let ArdSquare::Packed(cache) = slots.ard(*at)
        {
            let k = match node.own {
                Some(k) => k,
                None => {
                    let Some(gram) = bufs.first_mut() else {
                        return Err(too_few_buffers());
                    };
                    rbf.apply_from_sq_diff::<M, T>(cache, gram.as_mut(), Triangle::Lower)?;
                    gram.as_ref()
                }
            };
            rbf.contract_square_from_sq_diff(weight, k, cache, out, walk.fold)?;
            return Ok(if node.own.is_some() || node.value {
                lower_dot(weight, k)
            } else {
                0.0
            });
        }
        // Accurate ARD RBF: `∂k/∂θ_d = k · w_d (Δ_d)²`, so one Gram covers
        // every lengthscale and the sum is one matrix product. `FastApprox`
        // stays on the per-parameter loop: its derivative is the jet, not
        // `k` times the squared distance.
        if M::ACCURATE
            && matches!(self, Self::RbfArd(_))
            && let Some(value) = self.contract_ard_square::<M>(walk, weight, out, bufs, node)?
        {
            return Ok(value);
        }
        let Some(d_k) = bufs.first_mut() else {
            return Err(too_few_buffers());
        };
        let value = match (node.own, node.value) {
            (Some(k), _) => lower_dot(weight, k),
            (None, true) => {
                self.eval_gram::<M>(
                    walk.inputs,
                    d_k.as_mut(),
                    Triangle::Lower,
                    walk.scratch.as_mut(),
                    walk.nested,
                )?;
                lower_dot(weight, d_k.as_ref())
            }
            (None, false) => 0.0,
        };
        for (p, slot) in out.iter_mut().enumerate() {
            self.grad_gram::<M>(
                walk.inputs,
                d_k.as_mut(),
                p,
                Triangle::Lower,
                walk.scratch.as_mut(),
                walk.nested,
            )?;
            *slot = lower_dot(weight, d_k.as_ref());
        }
        Ok(value)
    }

    /// Accurate square ARD RBF: one Gram, then every lengthscale from `k`.
    fn contract_ard_square<M: crate::math::KernelMath>(
        &self,
        walk: &mut WeightedWalk<'_, '_, T, S>,
        weight: MatRef<'_, T>,
        out: &mut [f64],
        bufs: &mut [Mat<T>],
        node: Node<'_, T>,
    ) -> Result<Option<f64>, GprError> {
        let Self::RbfArd(leaf) = self else {
            return Ok(None);
        };
        // `apply_points_with` rejects a scratch that is not `n×n`. A bare ARD
        // leaf does not allocate that scratch, and this Gram does not read it.
        if node.own.is_none() {
            let Some(gram) = bufs.first_mut() else {
                return Err(too_few_buffers());
            };
            if let Some(cache) = walk.inputs.ard {
                leaf.apply_from_sq_diff::<M, T>(cache, gram.as_mut(), Triangle::Lower)?;
            } else {
                leaf.apply_math::<M, T>(walk.inputs.x, gram.as_mut(), Triangle::Lower)?;
            }
        }
        let own = node.own;
        let k = match own {
            Some(k) => k,
            None => bufs[0].as_ref(),
        };
        leaf.contract_square(walk.inputs.x, weight, k, walk.inputs.ard, out, walk.fold)?;
        let value = match (own, node.value) {
            (Some(k), _) => lower_dot(weight, k),
            (None, true) => lower_dot(weight, bufs[0].as_ref()),
            (None, false) => 0.0,
        };
        Ok(Some(value))
    }

    /// [`Self::weighted_cross_grads`] of the block `views` describes.
    // The block, the weight, the output, and three scratch kinds.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn weighted_cross_grads_views<M: crate::math::KernelMath>(
        &self,
        views: CrossViews<'_, T, S>,
        weight: MatRef<'_, T>,
        out: &mut [f64],
        bufs: &mut [Mat<T>],
        mut scratch: MatMut<'_, T>,
        nested: &mut [Mat<T>],
        jobs: &mut Vec<f64>,
    ) -> Result<(), GprError> {
        self.require_tree_columns(views.x1)?;
        self.require_tree_columns(views.x2)?;
        out.fill(0.0);
        let mut offset = 0;
        for (t, _) in self.keep_plan(0) {
            let count = t.num_params();
            t.cross_walk::<M>(
                views,
                weight,
                &mut out[offset..offset + count],
                bufs,
                scratch.as_mut(),
                nested,
                jobs,
                false,
            )?;
            offset += count;
        }
        Ok(())
    }

    /// `⟨weight, ∂K(x1, x2)/∂θ⟩` of one node. Returns `⟨weight, K⟩_F` when
    /// `want_value` is set.
    // Same arguments as [`Self::weighted_cross_grads`], plus the value flag.
    #[allow(clippy::too_many_arguments)]
    fn cross_walk<M: crate::math::KernelMath>(
        &self,
        views: CrossViews<'_, T, S>,
        weight: MatRef<'_, T>,
        out: &mut [f64],
        bufs: &mut [Mat<T>],
        mut scratch: MatMut<'_, T>,
        nested: &mut [Mat<T>],
        jobs: &mut Vec<f64>,
        want_value: bool,
    ) -> Result<f64, GprError> {
        match self {
            Self::Sum(terms) => {
                let mut offset = 0;
                let mut value = 0.0;
                for t in terms {
                    let count = t.num_params();
                    value += t.cross_walk::<M>(
                        views,
                        weight,
                        &mut out[offset..offset + count],
                        bufs,
                        scratch.as_mut(),
                        nested,
                        jobs,
                        want_value,
                    )?;
                    offset += count;
                }
                Ok(value)
            }
            Self::Product(terms) => self.cross_product::<M>(
                terms, views, weight, out, bufs, scratch, nested, jobs, want_value,
            ),
            Self::Constant(leaf) => {
                let value = leaf.constant() * rect_sum(weight);
                out[0] = value;
                Ok(value)
            }
            _ => self.cross_leaf::<M>(views, weight, out, bufs, scratch, nested, jobs, want_value),
        }
    }

    // The product's terms plus the arguments of [`Self::cross_walk`].
    #[allow(clippy::too_many_arguments)]
    fn cross_product<M: crate::math::KernelMath>(
        &self,
        terms: &[Self],
        views: CrossViews<'_, T, S>,
        weight: MatRef<'_, T>,
        out: &mut [f64],
        bufs: &mut [Mat<T>],
        mut scratch: MatMut<'_, T>,
        nested: &mut [Mat<T>],
        jobs: &mut Vec<f64>,
        want_value: bool,
    ) -> Result<f64, GprError> {
        let scale = Self::constant_scale(terms);
        let has_constant = terms.iter().any(|t| matches!(t, Self::Constant(_)));
        let need_value = want_value || has_constant;
        let varying = Self::varying_factors(terms);
        let value = match varying {
            0 => scale * rect_sum(weight),
            1 => {
                let mut offset = 0;
                let mut value = 0.0;
                for t in terms {
                    let count = t.num_params();
                    if !matches!(t, Self::Constant(_)) {
                        let slot = &mut out[offset..offset + count];
                        let inner = t.cross_walk::<M>(
                            views,
                            weight,
                            slot,
                            bufs,
                            scratch.as_mut(),
                            nested,
                            jobs,
                            need_value,
                        )?;
                        for g in slot.iter_mut() {
                            *g *= scale;
                        }
                        value = scale * inner;
                    }
                    offset += count;
                }
                value
            }
            _ => {
                if bufs.len() < varying + 1 {
                    return Err(too_few_buffers());
                }
                let (grams, rest) = bufs.split_at_mut(varying);
                let Some((handed, deeper)) = rest.split_first_mut() else {
                    return Err(too_few_buffers());
                };
                let factors = terms.iter().filter(|t| !matches!(t, Self::Constant(_)));
                for (factor, gram) in factors.zip(grams.iter_mut()) {
                    factor.apply_cross_mixed::<M>(
                        views,
                        gram.as_mut(),
                        scratch.as_mut(),
                        nested,
                    )?;
                }
                let mut offset = 0;
                let mut c = 0;
                let mut value = 0.0;
                for t in terms {
                    let count = t.num_params();
                    if matches!(t, Self::Constant(_)) {
                        offset += count;
                        continue;
                    }
                    write_handed_rect(handed.as_mut(), weight, scale, grams, c);
                    let leaf = !matches!(t, Self::Sum(_) | Self::Product(_));
                    let inner = t.cross_walk::<M>(
                        views,
                        handed.as_ref(),
                        &mut out[offset..offset + count],
                        deeper,
                        scratch.as_mut(),
                        nested,
                        jobs,
                        need_value && c == 0 && leaf,
                    )?;
                    if need_value && c == 0 {
                        value = if leaf {
                            inner
                        } else {
                            rect_dot(handed.as_ref(), grams[c].as_ref())
                        };
                    }
                    c += 1;
                    offset += count;
                }
                value
            }
        };
        let mut offset = 0;
        for t in terms {
            if matches!(t, Self::Constant(_)) {
                out[offset] = value;
            }
            offset += t.num_params();
        }
        Ok(value)
    }

    // The arguments of [`Self::cross_walk`].
    #[allow(clippy::too_many_arguments)]
    fn cross_leaf<M: crate::math::KernelMath>(
        &self,
        views: CrossViews<'_, T, S>,
        weight: MatRef<'_, T>,
        out: &mut [f64],
        bufs: &mut [Mat<T>],
        mut scratch: MatMut<'_, T>,
        nested: &mut [Mat<T>],
        jobs: &mut Vec<f64>,
        want_value: bool,
    ) -> Result<f64, GprError> {
        if let Self::RbfArd(leaf) = self {
            return leaf.contract_cross::<M, T>(views.x1, views.x2, weight, out, jobs, want_value);
        }
        let Some(d_k) = bufs.first_mut() else {
            return Err(too_few_buffers());
        };
        let value = if want_value {
            self.apply_cross_mixed::<M>(views, d_k.as_mut(), scratch.as_mut(), nested)?;
            rect_dot(weight, d_k.as_ref())
        } else {
            0.0
        };
        for (p, slot) in out.iter_mut().enumerate() {
            self.grad_cross_views::<M>(views, d_k.as_mut(), p, scratch.as_mut(), nested)?;
            *slot = rect_dot(weight, d_k.as_ref());
        }
        Ok(value)
    }

    /// Writes `Σ_i ∂k(x_i, x_i)/∂θ_p` for every parameter into `out`.
    ///
    /// A product reads each non-constant factor's diagonal once and scales
    /// the factor that owns `θ`. `out` is replaced. `accum` is scratch.
    pub(crate) fn weighted_diag_sums<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, T>,
        out: &mut [f64],
        accum: &mut DiagAccum<T>,
    ) -> Result<(), GprError> {
        self.require_tree_columns(x)?;
        out.fill(0.0);
        let mut cursor = 0;
        let mut offset = 0;
        for (t, _) in self.keep_plan(0) {
            let count = t.num_params();
            t.diag_accum::<M>(x, 0, &mut out[offset..offset + count], accum, &mut cursor)?;
            offset += count;
        }
        Ok(())
    }

    /// One node of [`Self::weighted_diag_sums`]. `depth == 0` scales by ones.
    /// A deeper node reads its scale at `accum.scale[(depth - 1) * n..]`.
    fn diag_accum<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, T>,
        depth: usize,
        out: &mut [f64],
        accum: &mut DiagAccum<T>,
        cursor: &mut usize,
    ) -> Result<(), GprError> {
        match self {
            Self::Sum(terms) => {
                let mut offset = 0;
                for t in terms {
                    let count = t.num_params();
                    t.diag_accum::<M>(x, depth, &mut out[offset..offset + count], accum, cursor)?;
                    offset += count;
                }
                Ok(())
            }
            Self::Product(terms) => self.diag_product::<M>(terms, x, depth, out, accum, cursor),
            _ => self.diag_leaf::<M>(x, depth, out, accum),
        }
    }

    fn diag_leaf<M: crate::math::KernelMath>(
        &self,
        x: MatRef<'_, T>,
        depth: usize,
        out: &mut [f64],
        accum: &mut DiagAccum<T>,
    ) -> Result<(), GprError> {
        let n = x.nrows();
        accum.fit_row(n);
        for (p, slot) in out.iter_mut().enumerate() {
            self.grad_diag_points::<M>(x, &mut accum.row[..n], p)?;
            let mut sum = 0.0;
            for i in 0..n {
                sum += accum.scale_at(depth, n, i) * accum.row[i].to_f64();
            }
            *slot = sum;
        }
        Ok(())
    }

    fn diag_product<M: crate::math::KernelMath>(
        &self,
        terms: &[Self],
        x: MatRef<'_, T>,
        depth: usize,
        out: &mut [f64],
        accum: &mut DiagAccum<T>,
        cursor: &mut usize,
    ) -> Result<(), GprError> {
        let n = x.nrows();
        let saved = *cursor;
        let n_varying = terms
            .iter()
            .filter(|t| !matches!(t, Self::Constant(_)))
            .count();
        accum.fit_flat(saved + n_varying * n);
        let mut k = 0;
        for t in terms {
            if matches!(t, Self::Constant(_)) {
                continue;
            }
            let start = saved + k * n;
            t.fill_diag_rows(x, &mut accum.flat[start..start + n])?;
            k += 1;
        }
        *cursor = saved + n_varying * n;
        accum.fit_scale((depth + 1) * n);
        let mut offset = 0;
        k = 0;
        for t in terms {
            let count = t.num_params();
            if !matches!(t, Self::Constant(_)) {
                let frame = depth * n;
                for i in 0..n {
                    let mut v =
                        T::from_f64(Self::constant_scale(terms) * accum.scale_at(depth, n, i));
                    for s in 0..n_varying {
                        if s != k {
                            v *= accum.flat[saved + s * n + i];
                        }
                    }
                    accum.scale[frame + i] = v;
                }
                t.diag_accum::<M>(
                    x,
                    depth + 1,
                    &mut out[offset..offset + count],
                    accum,
                    cursor,
                )?;
                k += 1;
            }
            offset += count;
        }
        offset = 0;
        for (index, t) in terms.iter().enumerate() {
            let count = t.num_params();
            if matches!(t, Self::Constant(_)) {
                let others = Self::constant_scale_except(terms, index);
                accum.fit_row(n);
                for (p, slot) in out[offset..offset + count].iter_mut().enumerate() {
                    t.grad_diag_points::<M>(x, &mut accum.row[..n], p)?;
                    let mut sum = 0.0;
                    for i in 0..n {
                        let mut v = accum.row[i].to_f64() * others * accum.scale_at(depth, n, i);
                        for s in 0..n_varying {
                            v *= accum.flat[saved + s * n + i].to_f64();
                        }
                        sum += v;
                    }
                    *slot = sum;
                }
            }
            offset += count;
        }
        *cursor = saved;
        Ok(())
    }

    /// `∏ c` over constant factors other than `skip`.
    fn constant_scale_except(terms: &[Self], skip: usize) -> f64 {
        terms
            .iter()
            .enumerate()
            .map(|(i, t)| match t {
                Self::Constant(leaf) if i != skip => leaf.constant(),
                _ => 1.0,
            })
            .product()
    }
}

/// What a node of the walk is handed besides its weight.
#[derive(Clone, Copy)]
struct Node<'g, T> {
    /// A product's kept factor Grams, in factor order.
    grams: Option<&'g [Mat<T>]>,
    /// This node's own Gram at the same `θ`, when a product evaluated it.
    own: Option<MatRef<'g, T>>,
    /// Whether the caller needs `⟨weight, K⟩_F`.
    value: bool,
}

/// `out = scale · ∏ grams` (or `out += …` when `accumulate`) on the lower
/// triangle, on the Rayon pool.
fn write_scaled_product<T: KernelScalar>(
    out: MatMut<'_, T>,
    scale: f64,
    grams: &[Mat<T>],
    accumulate: bool,
) {
    let scale = T::from_f64(scale);
    for_each_lower_col(out, &|col, mut rows| {
        for i in 0..rows.nrows() {
            let mut v = scale;
            for gram in grams {
                v *= gram[(col + i, col)];
            }
            rows[i] = if accumulate { rows[i] + v } else { v };
        }
    });
}

/// The weight handed to factor `c` of a product: `scale · weight ∘ ∏_{s≠c} grams[s]`
/// on the lower triangle, on the Rayon pool.
fn write_handed<T: KernelScalar>(
    out: MatMut<'_, T>,
    weight: MatRef<'_, T>,
    scale: f64,
    grams: &[Mat<T>],
    c: usize,
) {
    let scale = T::from_f64(scale);
    for_each_lower_col(out, &|col, mut rows| {
        for i in 0..rows.nrows() {
            let row = col + i;
            let mut v = weight[(row, col)] * scale;
            for (s, gram) in grams.iter().enumerate() {
                if s != c {
                    v *= gram[(row, col)];
                }
            }
            rows[i] = v;
        }
    });
}

fn copy_lower<T: KernelScalar>(out: MatMut<'_, T>, src: MatRef<'_, T>) {
    for_each_lower_col(out, &|col, mut rows| {
        for i in 0..rows.nrows() {
            rows[i] = src[(col + i, col)];
        }
    });
}

/// `⟨a, b⟩_F` of two symmetric matrices from their lower triangles, in
/// `f64`, on the Rayon pool (column sums added in column order).
fn lower_dot<T: KernelScalar>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> f64 {
    lower_fold(a.nrows(), |col, rows| {
        let mut off = 0.0;
        for row in rows {
            off += a[(row, col)].to_f64() * b[(row, col)].to_f64();
        }
        a[(col, col)].to_f64() * b[(col, col)].to_f64() + 2.0 * off
    })
}

/// `Σ_ij a_ij` of a symmetric matrix from its lower triangle, in `f64`,
/// on the Rayon pool (column sums added in column order).
fn lower_sum<T: KernelScalar>(a: MatRef<'_, T>) -> f64 {
    lower_fold(a.nrows(), |col, rows| {
        let mut off = 0.0;
        for row in rows {
            off += a[(row, col)].to_f64();
        }
        a[(col, col)].to_f64() + 2.0 * off
    })
}

/// Sums `column(col, col + 1..n)` (one column of the lower triangle, its
/// strict part as the row range) over every column.
fn lower_fold(n: usize, column: impl Fn(usize, std::ops::Range<usize>) -> f64 + Sync) -> f64 {
    lower_fold_infallible(
        n,
        &|start, end| {
            let mut sum = 0.0;
            for col in start..end {
                sum += column(col, col + 1..n);
            }
            sum
        },
        &|a, b| a + b,
    )
}

/// The weight handed to factor `c` of a product, on the full rectangle:
/// `scale · weight ∘ ∏_{s≠c} grams[s]`.
fn write_handed_rect<T: KernelScalar>(
    mut out: MatMut<'_, T>,
    weight: MatRef<'_, T>,
    scale: f64,
    grams: &[Mat<T>],
    c: usize,
) {
    let scale = T::from_f64(scale);
    let (rows, cols) = (out.nrows(), out.ncols());
    for col in 0..cols {
        for row in 0..rows {
            let mut v = weight[(row, col)] * scale;
            for (s, gram) in grams.iter().enumerate() {
                if s != c {
                    v *= gram[(row, col)];
                }
            }
            out[(row, col)] = v;
        }
    }
}

/// `⟨a, b⟩_F` of two full rectangles, summed in `f64`.
fn rect_dot<T: KernelScalar>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> f64 {
    let mut sum = 0.0;
    for col in 0..a.ncols() {
        for row in 0..a.nrows() {
            sum += a[(row, col)].to_f64() * b[(row, col)].to_f64();
        }
    }
    sum
}

/// `Σ_ij a_ij` of a full rectangle, summed in `f64`.
fn rect_sum<T: KernelScalar>(a: MatRef<'_, T>) -> f64 {
    let mut sum = 0.0;
    for col in 0..a.ncols() {
        for row in 0..a.nrows() {
            sum += a[(row, col)].to_f64();
        }
    }
    sum
}

/// Scratch for [`CompiledKernel::weighted_diag_sums`].
///
/// `flat` holds factor diagonals, `row` one parameter's `∂k_ii`, and
/// `scale` one length-`n` frame per product depth. It starts empty and grows
/// to the largest `n` asked for.
pub(crate) struct DiagAccum<T> {
    flat: Vec<T>,
    row: Vec<T>,
    scale: Vec<T>,
}

impl<T> DiagAccum<T> {
    pub(crate) fn new() -> Self {
        Self {
            flat: Vec::new(),
            row: Vec::new(),
            scale: Vec::new(),
        }
    }
}

impl<T: KernelScalar> DiagAccum<T> {
    fn fit_flat(&mut self, n: usize) {
        if self.flat.len() < n {
            self.flat.resize(n, T::from_f64(0.0));
        }
    }

    fn fit_row(&mut self, n: usize) {
        if self.row.len() < n {
            self.row.resize(n, T::from_f64(0.0));
        }
    }

    fn fit_scale(&mut self, n: usize) {
        if self.scale.len() < n {
            self.scale.resize(n, T::from_f64(0.0));
        }
    }

    /// The per-point scale at `depth`. Depth 0 is ones.
    fn scale_at(&self, depth: usize, n: usize, i: usize) -> f64 {
        if depth == 0 {
            1.0
        } else {
            self.scale[(depth - 1) * n + i].to_f64()
        }
    }
}

fn too_few_buffers() -> GprError {
    GprError::WorkspaceTooSmall
}

/// A leaf with a one-pass weighted walk on a distance matrix.
enum FastLeaf<'k> {
    Periodic(&'k crate::kernel::PeriodicKernel),
    RationalQuadratic(&'k crate::kernel::RationalQuadraticKernel),
    Rbf(&'k crate::kernel::RbfKernel),
    None,
}

impl<'k> FastLeaf<'k> {
    fn from_scalar<T: KernelScalar>(leaf: &'k ScalarLeaf<T>) -> Self {
        match leaf {
            ScalarLeaf::Periodic(leaf) => Self::Periodic(leaf),
            ScalarLeaf::RationalQuadratic(leaf) => Self::RationalQuadratic(leaf),
            ScalarLeaf::Rbf(leaf) => Self::Rbf(leaf),
            ScalarLeaf::Matern(_) | ScalarLeaf::Custom(_) => Self::None,
        }
    }
}

/// Coordinate entry points of a coordinate tree.
impl<T: KernelScalar> CompiledKernel<T> {
    /// Writes `⟨weight, ∂K(x1, x2)/∂θ_p⟩_F` for every parameter into `out`.
    ///
    /// The rectangle is full, not a triangle. A product evaluates each
    /// non-constant factor once and hands the others down in the weight, as
    /// [`Self::weighted_grads`] does for a square Gram. `out` is replaced.
    /// `bufs` holds [`Self::contraction_buffers`] matrices of `weight`'s shape.
    // The two point sets, the weight, the output, and three scratch kinds.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn weighted_cross_grads<M: crate::math::KernelMath>(
        &self,
        x1: MatRef<'_, T>,
        x2: MatRef<'_, T>,
        weight: MatRef<'_, T>,
        out: &mut [f64],
        bufs: &mut [Mat<T>],
        scratch: MatMut<'_, T>,
        nested: &mut [Mat<T>],
        jobs: &mut Vec<f64>,
    ) -> Result<(), GprError> {
        self.weighted_cross_grads_views::<M>(
            CrossViews::points(x1, x2),
            weight,
            out,
            bufs,
            scratch,
            nested,
            jobs,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{CompiledKernel, WeightedWalk};
    use crate::kernel::compiled::gram::GramInputs;
    use crate::kernel::{
        ConstantKernel, KernelSpec, PeriodicKernel, RationalQuadraticKernel, RbfArdKernel,
        RbfKernel, Triangle,
    };
    use crate::math::{Accurate, FastApprox, KernelMath};
    use crate::test_check::sq_dist_1d;
    use faer::Mat;

    /// Under one root sum: two products that can keep their Grams, one
    /// product with a single non-constant factor, and a bare leaf.
    fn mixed_root_sum() -> CompiledKernel<f64> {
        let c = |v| KernelSpec::from(ConstantKernel::new(v).expect("valid"));
        let rbf = |l| KernelSpec::from(RbfKernel::new(l).expect("valid"));
        let periodic = |l, p| KernelSpec::from(PeriodicKernel::new(l, p).expect("valid"));
        let rq = KernelSpec::from(RationalQuadraticKernel::new(0.6, 0.8).expect("valid"));
        (c(1.3) * rbf(1.7) * periodic(0.9, 1.4)
            + rbf(0.6) * periodic(1.1, 2.3)
            + c(0.4) * rq
            + rbf(2.2))
        .compile()
    }

    fn lower_bits(m: &Mat<f64>) -> Vec<u64> {
        let n = m.nrows();
        (0..n)
            .flat_map(|col| (col..n).map(move |row| (row, col)))
            .map(|(row, col)| m[(row, col)].to_bits())
            .collect()
    }

    /// `K` and every `⟨W, ∂K/∂θ⟩` when the first `products` products keep
    /// their Grams.
    fn keep_and_walk<M: KernelMath>(
        compiled: &CompiledKernel<f64>,
        products: usize,
    ) -> (Vec<u64>, Vec<u64>) {
        let xs = [0.0, 0.8, 1.7, 0.2, -0.4, 0.9, 1.1, 0.3, 2.0];
        let n = xs.len();
        let x = Mat::from_fn(n, 1, |i, _| xs[i]);
        let dist = sq_dist_1d(&xs);
        let inputs = GramInputs {
            x: x.as_ref(),
            dist: Some(dist.as_ref()),
            ard: None,
            slots: (),
        };
        let weight = Mat::from_fn(n, n, |i, j| {
            let (a, b) = (i.max(j) as f64, i.min(j) as f64);
            0.3 * a - 0.7 * b + 0.05 * a * b - 0.4
        });
        let zeros = |count: usize| (0..count).map(|_| Mat::zeros(n, n)).collect::<Vec<_>>();
        let mut kept = zeros(compiled.kept_buffers(products));
        let mut bufs = zeros(compiled.weighted_buffers(products) - kept.len());
        let (mut k, mut scratch, mut term) = (Mat::zeros(n, n), Mat::zeros(n, n), Mat::zeros(n, n));
        let mut nested = Vec::new();
        compiled
            .eval_gram_keeping::<M>(
                inputs,
                k.as_mut(),
                scratch.as_mut(),
                term.as_mut(),
                &mut nested,
                &mut kept,
                products,
            )
            .expect("gram");
        let mut grads = vec![0.0; compiled.num_params()];
        let mut fold = Vec::new();
        let mut walk = WeightedWalk {
            inputs,
            scratch: scratch.as_mut(),
            nested: &mut nested,
            kept: &kept,
            kept_products: products,
            fold: &mut fold,
        };
        compiled
            .weighted_grads::<M>(&mut walk, weight.as_ref(), &mut grads, &mut bufs)
            .expect("walk");
        (lower_bits(&k), grads.iter().map(|g| g.to_bits()).collect())
    }

    fn every_keep_count_gives_the_same_bits<M: KernelMath>() {
        let compiled = mixed_root_sum();
        let candidates = compiled
            .keep_plan(usize::MAX)
            .filter(|(_, slot)| slot.is_some())
            .count();
        assert_eq!(candidates, 2);
        let (k0, g0) = keep_and_walk::<M>(&compiled, 0);
        let n = 9;
        let mut plain = Mat::zeros(n, n);
        let xs = [0.0, 0.8, 1.7, 0.2, -0.4, 0.9, 1.1, 0.3, 2.0];
        let x = Mat::from_fn(n, 1, |i, _| xs[i]);
        let dist = sq_dist_1d(&xs);
        compiled
            .eval_gram::<M>(
                GramInputs {
                    x: x.as_ref(),
                    dist: Some(dist.as_ref()),
                    ard: None,
                    slots: (),
                },
                plain.as_mut(),
                Triangle::Lower,
                Mat::zeros(n, n).as_mut(),
                &mut Vec::new(),
            )
            .expect("gram");
        assert_eq!(k0, lower_bits(&plain));
        for products in 1..=candidates {
            let (k, g) = keep_and_walk::<M>(&compiled, products);
            assert_eq!(k, k0, "K with {products} kept");
            assert_eq!(g, g0, "gradient with {products} kept");
        }
    }

    #[test]
    fn keeping_any_number_of_products_gives_the_same_bits() {
        every_keep_count_gives_the_same_bits::<Accurate>();
        every_keep_count_gives_the_same_bits::<FastApprox>();
    }

    fn assert_rel(got: &[f64], expect: &[f64]) {
        assert_eq!(got.len(), expect.len());
        for (i, (g, e)) in got.iter().zip(expect).enumerate() {
            let scale = e.abs().max(1.0);
            assert!(
                (g - e).abs() <= 1e-8 * scale,
                "i={i} walk={g} per-param={e}"
            );
        }
    }

    /// Constant × RBF × Periodic, plus an RBF: the square, rectangular, and
    /// diagonal contractions match one `∂K/∂θ` per parameter.
    fn contractions_match_per_parameter<M: KernelMath>() {
        let c = |v| KernelSpec::from(ConstantKernel::new(v).expect("c"));
        let rbf = |l| KernelSpec::from(RbfKernel::new(l).expect("ell"));
        let periodic = |l, p| KernelSpec::from(PeriodicKernel::new(l, p).expect("periodic"));
        let compiled = (c(1.3) * rbf(1.7) * periodic(0.9, 1.4) + rbf(0.6)).compile();
        let xz = [0.2, 0.9, 1.4];
        let xx = [0.0, 0.5, 1.1, 1.8, 2.4];
        let z = Mat::from_fn(xz.len(), 1, |i, _| xz[i]);
        let x = Mat::from_fn(xx.len(), 1, |i, _| xx[i]);
        let (m, n) = (z.nrows(), x.nrows());
        let raw = Mat::from_fn(m, m, |i, j| 0.2 * (i + 1) as f64 - 0.15 * (j + 1) as f64);
        let weight_zz = Mat::from_fn(m, m, |i, j| 0.5 * (raw[(i, j)] + raw[(j, i)]));
        let weight_zx = Mat::from_fn(m, n, |i, j| 0.3 * (i + 1) as f64 - 0.2 * j as f64 + 0.05);
        let dist = sq_dist_1d(&xz);
        let inputs = GramInputs {
            x: z.as_ref(),
            dist: Some(dist.as_ref()),
            ard: None,
            slots: (),
        };
        let n_params = compiled.num_params();
        let mut got = vec![0.0; n_params];
        let mut expect = vec![0.0; n_params];
        let mut scratch = Mat::zeros(m, m);
        let mut nested = Vec::new();
        let mut bufs = vec![Mat::zeros(m, m); compiled.contraction_buffers()];
        {
            let mut fold = Vec::new();
            let mut walk = WeightedWalk {
                inputs,
                scratch: scratch.as_mut(),
                nested: &mut nested,
                kept: &[],
                kept_products: 0,
                fold: &mut fold,
            };
            compiled
                .weighted_grads::<M>(&mut walk, weight_zz.as_ref(), &mut got, &mut bufs)
                .expect("square");
        }
        let mut d_k = Mat::zeros(m, m);
        for (p, slot) in expect.iter_mut().enumerate() {
            compiled
                .grad_gram::<M>(
                    inputs,
                    d_k.as_mut(),
                    p,
                    Triangle::Lower,
                    scratch.as_mut(),
                    &mut nested,
                )
                .expect("grad");
            *slot = super::lower_dot(weight_zz.as_ref(), d_k.as_ref());
        }
        assert_rel(&got, &expect);

        let mut cross_scratch = Mat::zeros(m, n);
        let mut cross_nested = Vec::new();
        crate::kernel::ensure_nested_levels(&mut cross_nested, &compiled, m, n);
        let mut cross_bufs = vec![Mat::zeros(m, n); compiled.contraction_buffers()];
        compiled
            .weighted_cross_grads::<M>(
                z.as_ref(),
                x.as_ref(),
                weight_zx.as_ref(),
                &mut got,
                &mut cross_bufs,
                cross_scratch.as_mut(),
                &mut cross_nested,
                &mut Vec::new(),
            )
            .expect("cross");
        let mut d_cross = Mat::zeros(m, n);
        for (p, slot) in expect.iter_mut().enumerate() {
            compiled
                .grad_cross_points_with::<M>(
                    z.as_ref(),
                    x.as_ref(),
                    d_cross.as_mut(),
                    p,
                    cross_scratch.as_mut(),
                    &mut cross_nested,
                )
                .expect("grad cross");
            *slot = super::rect_dot(weight_zx.as_ref(), d_cross.as_ref());
        }
        assert_rel(&got, &expect);

        let mut accum = super::DiagAccum::new();
        compiled
            .weighted_diag_sums::<M>(x.as_ref(), &mut got, &mut accum)
            .expect("diag");
        let mut row = vec![0.0; n];
        for (p, slot) in expect.iter_mut().enumerate() {
            compiled
                .grad_diag_points::<M>(x.as_ref(), &mut row, p)
                .expect("grad diag");
            *slot = row.iter().sum();
        }
        assert_rel(&got, &expect);
    }

    #[test]
    fn cross_and_diag_contractions_match_per_parameter() {
        contractions_match_per_parameter::<Accurate>();
        contractions_match_per_parameter::<FastApprox>();
    }

    /// Constant × ARD RBF, and a bare ARD RBF. The lengthscale sum is one
    /// matrix product of the lower triangle.
    fn ard_square_contraction_matches_per_parameter<M: KernelMath>() {
        let c = |v| KernelSpec::from(ConstantKernel::new(v).expect("c"));
        let ard = |ls: &[f64]| KernelSpec::from(RbfArdKernel::new(ls).expect("ard"));
        for spec in [c(1.3) * ard(&[0.7, 1.4, 2.2]), ard(&[0.5, 0.9, 1.1])] {
            let compiled = spec.compile();
            let d = 3;
            let n = 20;
            let x = Mat::from_fn(n, d, |i, j| 0.15 * i as f64 - 0.11 * j as f64 - 0.4);
            let inputs = GramInputs::points(x.as_ref());
            let weight = Mat::from_fn(n, n, |i, j| 0.3 * (i + 1) as f64 - 0.2 * j as f64 + 0.05);
            let mut got = vec![0.0; compiled.num_params()];
            let mut expect = vec![0.0; compiled.num_params()];
            let mut scratch = Mat::zeros(n, n);
            let mut nested = Vec::new();
            crate::kernel::ensure_nested_levels(&mut nested, &compiled, n, n);
            let mut bufs = vec![Mat::zeros(n, n); compiled.weighted_buffers(0)];
            let mut fold = Vec::new();
            let mut walk = WeightedWalk {
                inputs,
                scratch: scratch.as_mut(),
                nested: &mut nested,
                kept: &[],
                kept_products: 0,
                fold: &mut fold,
            };
            compiled
                .weighted_grads::<M>(&mut walk, weight.as_ref(), &mut got, &mut bufs)
                .expect("walk");
            let mut d_k = Mat::zeros(n, n);
            for (p, slot) in expect.iter_mut().enumerate() {
                compiled
                    .grad_gram::<M>(
                        inputs,
                        d_k.as_mut(),
                        p,
                        Triangle::Lower,
                        scratch.as_mut(),
                        &mut nested,
                    )
                    .expect("grad");
                *slot = super::lower_dot(weight.as_ref(), d_k.as_ref());
            }
            for (i, (g, e)) in got.iter().zip(&expect).enumerate() {
                let scale = e.abs().max(1.0);
                assert!(
                    (g - e).abs() <= 1e-7 * scale,
                    "i={i} walk={g} per-param={e}"
                );
            }
        }
    }

    #[test]
    fn ard_square_contraction_matches_one_parameter_at_a_time() {
        ard_square_contraction_matches_per_parameter::<Accurate>();
        ard_square_contraction_matches_per_parameter::<FastApprox>();
    }

    /// Constant × ARD RBF, plus an ARD RBF. The rectangle is wider than one
    /// column batch and the column length is not a multiple of four, so the
    /// one-exp contraction covers a second batch and a scalar tail.
    fn ard_cross_contraction_matches_per_parameter<M: KernelMath>() {
        let c = |v| KernelSpec::from(ConstantKernel::new(v).expect("c"));
        let ard = |ls: &[f64]| KernelSpec::from(RbfArdKernel::new(ls).expect("ard"));
        let compiled = (c(1.3) * ard(&[0.7, 1.4, 2.2]) + ard(&[0.5, 0.9, 1.1])).compile();
        let d = 3;
        let z_rows = [
            [0.2, -0.4, 0.7],
            [1.1, 0.3, -0.2],
            [-0.5, 0.8, 1.4],
            [0.0, 0.2, 0.9],
            [1.5, -1.0, 0.4],
            [0.6, 0.1, -0.8],
        ];
        let mut x_rows = [[0.0; 3]; 20];
        for (i, row) in x_rows.iter_mut().enumerate() {
            row[0] = 0.15 * i as f64 - 0.4;
            row[1] = 0.07 * (i as f64 - 3.0);
            row[2] = -0.11 * i as f64 + 0.5;
        }
        let z = Mat::from_fn(z_rows.len(), d, |i, j| z_rows[i][j]);
        let x = Mat::from_fn(x_rows.len(), d, |i, j| x_rows[i][j]);
        let (m, n) = (z.nrows(), x.nrows());
        let weight = Mat::from_fn(m, n, |i, j| 0.3 * (i + 1) as f64 - 0.2 * j as f64 + 0.05);
        let mut got = vec![0.0; compiled.num_params()];
        let mut expect = vec![0.0; compiled.num_params()];
        let mut scratch = Mat::zeros(m, n);
        let mut nested = Vec::new();
        crate::kernel::ensure_nested_levels(&mut nested, &compiled, m, n);
        let mut bufs = vec![Mat::zeros(m, n); compiled.contraction_buffers()];
        compiled
            .weighted_cross_grads::<M>(
                z.as_ref(),
                x.as_ref(),
                weight.as_ref(),
                &mut got,
                &mut bufs,
                scratch.as_mut(),
                &mut nested,
                &mut Vec::new(),
            )
            .expect("cross");
        let mut d_cross = Mat::zeros(m, n);
        for (p, slot) in expect.iter_mut().enumerate() {
            compiled
                .grad_cross_points_with::<M>(
                    z.as_ref(),
                    x.as_ref(),
                    d_cross.as_mut(),
                    p,
                    scratch.as_mut(),
                    &mut nested,
                )
                .expect("grad cross");
            *slot = super::rect_dot(weight.as_ref(), d_cross.as_ref());
        }
        for (i, (g, e)) in got.iter().zip(&expect).enumerate() {
            let scale = e.abs().max(1.0);
            // The lengthscale sum is one matrix product. `rect_dot` is one
            // running total, so the two sums differ in the last digits.
            assert!(
                (g - e).abs() <= 1e-7 * scale,
                "i={i} walk={g} per-param={e}"
            );
        }
    }

    #[test]
    fn ard_cross_contraction_matches_one_parameter_at_a_time() {
        ard_cross_contraction_matches_per_parameter::<Accurate>();
        ard_cross_contraction_matches_per_parameter::<FastApprox>();
    }
}
