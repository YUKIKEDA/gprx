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
//! transcendental functions. Every matrix is the lower triangle of a
//! symmetric `n × n`.

use super::gram::GramInputs;
use super::{CompiledKernel, add_triangle};
use crate::error::GprError;
use crate::kernel::dist::{par_lower_blocks, par_lower_fold, worker_count};
use crate::kernel::{KernelScalar, Triangle};
use faer::{Mat, MatMut, MatRef};

/// Where the walk is: the inputs, the scratch it shares, and the kept
/// factor Grams with the number of products allowed to read them.
pub(crate) struct WeightedWalk<'a, 'b, T> {
    pub(crate) inputs: GramInputs<'a, T>,
    pub(crate) scratch: MatMut<'b, T>,
    pub(crate) nested: &'b mut Vec<Mat<T>>,
    /// The Grams [`CompiledKernel::eval_gram_keeping`] wrote, in walk order.
    pub(crate) kept: &'b [Mat<T>],
    /// How many products read `kept` ([`CompiledKernel::kept_products`]).
    pub(crate) kept_products: usize,
}

impl<T: KernelScalar> CompiledKernel<T> {
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

    /// The products whose factor Grams the factor step can keep, in walk
    /// order: the root itself, or each term of a root sum, when it has two
    /// or more non-constant factors. Sums are flattened when the tree is
    /// compiled, so no other product is reached through sums only.
    fn keep_candidates(&self) -> impl Iterator<Item = &[Self]> {
        let roots: &[Self] = match self {
            Self::Sum(terms) => terms,
            other => std::slice::from_ref(other),
        };
        roots.iter().filter_map(|t| match t {
            Self::Product(terms) if Self::varying_factors(terms) >= 2 => Some(terms.as_slice()),
            _ => None,
        })
    }

    /// `n×n` Grams kept for the first `products` candidates.
    fn kept_grams(&self, products: usize) -> usize {
        self.keep_candidates()
            .take(products)
            .map(Self::varying_factors)
            .sum()
    }

    /// How many products keep their factor Grams: the most for which the
    /// kept Grams and the walk's own buffers stay within
    /// [`Self::unkept_buffers`], the buffers a walk that keeps nothing and
    /// evaluates every factor (constants included) would take.
    pub(crate) fn kept_products(&self) -> usize {
        let budget = self.unkept_buffers();
        let candidates = self.keep_candidates().count();
        (0..=candidates)
            .rev()
            .find(|&p| self.kept_grams(p) + self.walk_buffers(p) <= budget)
            .unwrap_or(0)
    }

    /// The `n × n` buffers of the joint gradient: the kept Grams first
    /// (see [`Self::kept_products`]), then the walk's own.
    pub(crate) fn weighted_buffers(&self) -> usize {
        let products = self.kept_products();
        self.kept_grams(products) + self.walk_buffers(products)
    }

    /// The Grams [`Self::weighted_buffers`] starts with.
    pub(crate) fn kept_buffers(&self) -> usize {
        self.kept_grams(self.kept_products())
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

    /// The walk's own buffers when the first `kept` candidates read kept Grams.
    fn walk_buffers(&self, kept: usize) -> usize {
        match self {
            Self::Sum(terms) => {
                let mut seen = 0;
                terms
                    .iter()
                    .map(|t| {
                        let keeps = match t {
                            Self::Product(f) if Self::varying_factors(f) >= 2 => {
                                seen += 1;
                                seen <= kept
                            }
                            _ => false,
                        };
                        t.node_buffers(keeps)
                    })
                    .max()
                    .unwrap_or(0)
            }
            other => other.node_buffers(kept > 0),
        }
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
        inputs: GramInputs<'_, T>,
        mut out: MatMut<'_, T>,
        mut scratch: MatMut<'_, T>,
        mut term: MatMut<'_, T>,
        nested: &mut Vec<Mat<T>>,
        kept: &mut [Mat<T>],
        products: usize,
    ) -> Result<(), GprError> {
        if products == 0 {
            return self.eval_gram::<M>(inputs, out, Triangle::Lower, scratch, nested);
        }
        let n = out.nrows();
        match self {
            Self::Sum(terms) => {
                let mut seen = 0;
                let mut offset = 0;
                for (i, t) in terms.iter().enumerate() {
                    let keeps = match t {
                        Self::Product(f) if Self::varying_factors(f) >= 2 => {
                            seen += 1;
                            seen <= products
                        }
                        _ => false,
                    };
                    match t {
                        Self::Product(factors) if keeps => {
                            let count = Self::varying_factors(factors);
                            let grams = &mut kept[offset..offset + count];
                            offset += count;
                            Self::keep_factor_grams::<M>(
                                factors,
                                inputs,
                                grams,
                                scratch.as_mut(),
                                nested,
                            )?;
                            write_scaled_product(
                                out.as_mut(),
                                Self::constant_scale(factors),
                                grams,
                                i > 0,
                            );
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
                                copy_lower(out.as_mut(), term.as_ref(), n);
                            } else {
                                add_triangle(out.as_mut(), term.as_ref(), Triangle::Lower);
                            }
                        }
                    }
                }
                Ok(())
            }
            Self::Product(factors) if Self::varying_factors(factors) >= 2 => {
                let count = Self::varying_factors(factors);
                let grams = &mut kept[..count];
                Self::keep_factor_grams::<M>(factors, inputs, grams, scratch, nested)?;
                write_scaled_product(out, Self::constant_scale(factors), grams, false);
                Ok(())
            }
            _ => self.eval_gram::<M>(inputs, out, Triangle::Lower, scratch, nested),
        }
    }

    /// Evaluates every non-constant factor into its kept Gram.
    fn keep_factor_grams<M: crate::math::KernelMath>(
        factors: &[Self],
        inputs: GramInputs<'_, T>,
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
        walk: &mut WeightedWalk<'_, '_, T>,
        weight: MatRef<'_, T>,
        out: &mut [f64],
        bufs: &mut [Mat<T>],
    ) -> Result<(), GprError> {
        let products = walk.kept_products;
        let kept = walk.kept;
        match self {
            Self::Sum(terms) => {
                let mut seen = 0;
                let mut kept_offset = 0;
                let mut offset = 0;
                for t in terms {
                    let count = t.num_params();
                    let grams = match t {
                        Self::Product(f) if Self::varying_factors(f) >= 2 => {
                            seen += 1;
                            if seen <= products {
                                let varying = Self::varying_factors(f);
                                let grams = &kept[kept_offset..kept_offset + varying];
                                kept_offset += varying;
                                Some(grams)
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };
                    t.walk::<M>(
                        walk,
                        weight,
                        &mut out[offset..offset + count],
                        bufs,
                        Node {
                            grams,
                            own: None,
                            value: false,
                        },
                    )?;
                    offset += count;
                }
                Ok(())
            }
            _ => {
                let grams = match self {
                    Self::Product(f) if products > 0 && Self::varying_factors(f) >= 2 => {
                        Some(&kept[..Self::varying_factors(f)])
                    }
                    _ => None,
                };
                self.walk::<M>(
                    walk,
                    weight,
                    out,
                    bufs,
                    Node {
                        grams,
                        own: None,
                        value: false,
                    },
                )
                .map(|_| ())
            }
        }
    }

    /// One node of [`Self::weighted_grads`]. Returns `⟨weight, K⟩_F` when
    /// `node.value` asks for it (else whatever the pass found on the way).
    fn walk<M: crate::math::KernelMath>(
        &self,
        walk: &mut WeightedWalk<'_, '_, T>,
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
        walk: &mut WeightedWalk<'_, '_, T>,
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
        walk: &mut WeightedWalk<'_, '_, T>,
        weight: MatRef<'_, T>,
        out: &mut [f64],
        bufs: &mut [Mat<T>],
        node: Node<'_, T>,
    ) -> Result<f64, GprError> {
        // Leaves whose parameters share each entry's transcendental work:
        // one pass, no `∂K` matrix.
        match (self, walk.inputs.dist) {
            (Self::Periodic(leaf), Some(dist)) => {
                return leaf.weighted_grads_dist::<M, T>(dist, node.own, weight, out);
            }
            (Self::RationalQuadratic(leaf), Some(dist)) => {
                return leaf.weighted_grads_dist(dist, node.own, weight, out);
            }
            // From `k` when the leaf has it or must form it for `⟨V, K⟩`
            // anyway; otherwise the `∂K` path below, one `exp` per entry too.
            (Self::Rbf(leaf), Some(dist)) if M::ACCURATE && (node.own.is_some() || node.value) => {
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
            _ => {}
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
    let n = out.nrows();
    let scale = T::from_f64(scale);
    let _ = par_lower_blocks(out, worker_count(), &|start, mut part: MatMut<'_, T>| {
        for local in 0..part.ncols() {
            let col = start + local;
            for row in col..n {
                let mut v = scale;
                for gram in grams {
                    v *= gram[(row, col)];
                }
                part[(row, local)] = if accumulate {
                    part[(row, local)] + v
                } else {
                    v
                };
            }
        }
        Ok::<(), ()>(())
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
    let n = out.nrows();
    let scale = T::from_f64(scale);
    let _ = par_lower_blocks(out, worker_count(), &|start, mut part: MatMut<'_, T>| {
        for local in 0..part.ncols() {
            let col = start + local;
            for row in col..n {
                let mut v = weight[(row, col)] * scale;
                for (s, gram) in grams.iter().enumerate() {
                    if s != c {
                        v *= gram[(row, col)];
                    }
                }
                part[(row, local)] = v;
            }
        }
        Ok::<(), ()>(())
    });
}

fn copy_lower<T: KernelScalar>(out: MatMut<'_, T>, src: MatRef<'_, T>, n: usize) {
    let _ = par_lower_blocks(out, worker_count(), &|start, mut part: MatMut<'_, T>| {
        for local in 0..part.ncols() {
            let col = start + local;
            for row in col..n {
                part[(row, local)] = src[(row, col)];
            }
        }
        Ok::<(), ()>(())
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
    par_lower_fold(
        n,
        &|start, end| {
            let mut sum = 0.0;
            for col in start..end {
                sum += column(col, col + 1..n);
            }
            Ok::<f64, ()>(sum)
        },
        &|a, b| a + b,
    )
    .unwrap_or(0.0)
}

fn too_few_buffers() -> GprError {
    GprError::WorkspaceTooSmall
}
