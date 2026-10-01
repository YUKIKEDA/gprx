//! Every `⟨V, ∂K/∂θ⟩` of a kernel tree in one walk of the tree.
//!
//! The joint gradient of a fit needs `⟨W, ∂K/∂θ_i⟩_F` for every `θ_i`.
//! Writing `∂K/∂θ_i` one parameter at a time evaluates the other factors of
//! a product again for each parameter. This walk passes a weight matrix down
//! the tree instead: a sum hands its weight to every term unchanged, and a
//! product evaluates each factor's Gram once and hands factor `c` the weight
//! `V ∘ ∏_{s≠c} K_s`, since `⟨V, ∏_{s≠c} K_s ∘ ∂K_c⟩ = ⟨V ∘ ∏_{s≠c} K_s, ∂K_c⟩`.
//! A leaf writes its own `∂K/∂θ` once per parameter. Every matrix is the
//! lower triangle of a symmetric `n × n`.

use super::CompiledKernel;
use super::gram::GramInputs;
use crate::error::GprError;
use crate::kernel::{KernelScalar, Triangle};
use crate::linalg::frobenius_lower;
use faer::{Mat, MatMut, MatRef};

impl<T: KernelScalar> CompiledKernel<T> {
    /// The `n × n` buffers [`Self::weighted_grads`] uses at once.
    pub(crate) fn weighted_buffers(&self) -> usize {
        match self {
            Self::Sum(terms) => terms.iter().map(Self::weighted_buffers).max().unwrap_or(0),
            // Each factor's Gram, the weight handed down, and the deepest term.
            Self::Product(terms) => {
                terms.len() + 1 + terms.iter().map(Self::weighted_buffers).max().unwrap_or(0)
            }
            _ => 1,
        }
    }

    /// Writes `⟨weight, ∂K/∂θ_p⟩_F` for every parameter `p` of this tree
    /// into `out` (length [`Self::num_params`]), from the lower triangles.
    ///
    /// `bufs` holds at least [`Self::weighted_buffers`] matrices of
    /// `weight`'s shape; their contents are overwritten. `scratch` and
    /// `nested` are the buffers [`Self::eval_gram`] and [`Self::grad_gram`]
    /// take.
    pub(crate) fn weighted_grads<M: crate::math::KernelMath>(
        &self,
        inputs: GramInputs<'_, T>,
        weight: MatRef<'_, T>,
        out: &mut [f64],
        mut scratch: MatMut<'_, T>,
        nested: &mut Vec<Mat<T>>,
        bufs: &mut [Mat<T>],
    ) -> Result<(), GprError> {
        let n = weight.nrows();
        match self {
            Self::Sum(terms) => {
                let mut offset = 0;
                for term in terms {
                    let count = term.num_params();
                    term.weighted_grads::<M>(
                        inputs,
                        weight,
                        &mut out[offset..offset + count],
                        scratch.as_mut(),
                        nested,
                        bufs,
                    )?;
                    offset += count;
                }
                Ok(())
            }
            Self::Product(terms) => {
                let (grams, rest) = bufs.split_at_mut(terms.len());
                let Some((handed, deeper)) = rest.split_first_mut() else {
                    return Err(too_few_buffers());
                };
                for (term, gram) in terms.iter().zip(grams.iter_mut()) {
                    term.eval_gram::<M>(
                        inputs,
                        gram.as_mut(),
                        Triangle::Lower,
                        scratch.as_mut(),
                        nested,
                    )?;
                }
                let mut offset = 0;
                for (c, term) in terms.iter().enumerate() {
                    let count = term.num_params();
                    if count > 0 {
                        for col in 0..n {
                            for row in col..n {
                                let mut value = weight[(row, col)];
                                for (s, gram) in grams.iter().enumerate() {
                                    if s != c {
                                        value *= gram[(row, col)];
                                    }
                                }
                                handed[(row, col)] = value;
                            }
                        }
                        term.weighted_grads::<M>(
                            inputs,
                            handed.as_ref(),
                            &mut out[offset..offset + count],
                            scratch.as_mut(),
                            nested,
                            deeper,
                        )?;
                    }
                    offset += count;
                }
                Ok(())
            }
            _ => {
                let Some(d_k) = bufs.first_mut() else {
                    return Err(too_few_buffers());
                };
                for (p, slot) in out.iter_mut().enumerate() {
                    self.grad_gram::<M>(
                        inputs,
                        d_k.as_mut(),
                        p,
                        Triangle::Lower,
                        scratch.as_mut(),
                        nested,
                    )?;
                    *slot = frobenius_lower(weight, d_k.as_ref(), n).to_f64();
                }
                Ok(())
            }
        }
    }
}

fn too_few_buffers() -> GprError {
    GprError::WorkspaceTooSmall
}
