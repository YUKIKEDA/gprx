//! Double-double arithmetic (about 106 bits, `1e-32` relative) and the
//! reference values of a Sparse VFE system computed in it.
//!
//! A factor in `f64` of an ill-conditioned `K_mm` (a smooth kernel on a few
//! points; a near duplicate) is off the exact system by about its condition
//! number times `2.2e-16`, so the `f64` factor cannot say how far a model
//! that took another route is from the system. These values can: they
//! carry `K_mm` to `1e-30`, which a condition number up to `1e15` leaves
//! at `1e-15`.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// An unevaluated sum `hi + lo` with `|lo| ≤ ulp(hi) / 2`.
#[derive(Clone, Copy, Debug)]
pub struct Dd {
    hi: f64,
    lo: f64,
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

impl Dd {
    pub const ZERO: Dd = Dd { hi: 0.0, lo: 0.0 };
    pub const ONE: Dd = Dd { hi: 1.0, lo: 0.0 };

    pub fn new(x: f64) -> Dd {
        Dd { hi: x, lo: 0.0 }
    }

    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    /// `ln` to `f64` accuracy (one Newton step is not needed for a log
    /// determinant that is only compared to `1e-12`).
    pub fn ln_f64(self) -> f64 {
        self.hi.ln() + self.lo / self.hi
    }

    pub fn sqrt(self) -> Dd {
        let x = 1.0 / self.hi.sqrt();
        let ax = Dd::new(self.hi * x);
        let diff = self - ax * ax;
        ax + Dd::new(diff.hi * x * 0.5)
    }

    /// `exp(self)` for an argument of moderate size.
    pub fn exp(self) -> Dd {
        const LN2: Dd = Dd {
            hi: std::f64::consts::LN_2,
            lo: 2.319_046_813_846_299_6e-17,
        };
        let k = (self.hi / LN2.hi).round();
        // `x = k ln 2 + r`, `|r| ≤ 0.35`; `exp(r) = exp(r / 16)^16`.
        let r = (self - LN2 * Dd::new(k)) * Dd::new(1.0 / 16.0);
        let mut term = Dd::ONE;
        let mut sum = Dd::ONE;
        for i in 1..=17 {
            term = term * r / Dd::new(f64::from(i));
            sum = sum + term;
        }
        for _ in 0..4 {
            sum = sum * sum;
        }
        let scale = 2f64.powi(k as i32);
        Dd {
            hi: sum.hi * scale,
            lo: sum.lo * scale,
        }
    }
}

impl Add for Dd {
    type Output = Dd;
    fn add(self, o: Dd) -> Dd {
        let (s1, s2) = two_sum(self.hi, o.hi);
        let (t1, t2) = two_sum(self.lo, o.lo);
        let s2 = s2 + t1;
        let (s1, s2) = quick_two_sum(s1, s2);
        let s2 = s2 + t2;
        let (hi, lo) = quick_two_sum(s1, s2);
        Dd { hi, lo }
    }
}

impl Neg for Dd {
    type Output = Dd;
    fn neg(self) -> Dd {
        Dd {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl Sub for Dd {
    type Output = Dd;
    fn sub(self, o: Dd) -> Dd {
        self + (-o)
    }
}

impl Mul for Dd {
    type Output = Dd;
    fn mul(self, o: Dd) -> Dd {
        let (p1, p2) = two_prod(self.hi, o.hi);
        let p2 = p2 + (self.hi * o.lo + self.lo * o.hi);
        let (hi, lo) = quick_two_sum(p1, p2);
        Dd { hi, lo }
    }
}

impl Div for Dd {
    type Output = Dd;
    fn div(self, o: Dd) -> Dd {
        let q1 = self.hi / o.hi;
        let r = self - o * Dd::new(q1);
        let q2 = r.hi / o.hi;
        let r = r - o * Dd::new(q2);
        let q3 = r.hi / o.hi;
        let (hi, lo) = quick_two_sum(q1, q2);
        Dd { hi, lo } + Dd::new(q3)
    }
}

/// Lower Cholesky factor of the symmetric positive definite `n × n` matrix
/// `a` (row-major), or `None` when a pivot is not positive.
fn cholesky(a: &[Dd], n: usize) -> Option<Vec<Dd>> {
    let mut l = vec![Dd::ZERO; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut v = a[i * n + j];
            for k in 0..j {
                v = v - l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if v.hi <= 0.0 {
                    return None;
                }
                l[i * n + i] = v.sqrt();
            } else {
                l[i * n + j] = v / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Solves `L Lᵀ x = b` for the factor `l` of [`cholesky`].
fn solve(l: &[Dd], n: usize, b: &[Dd]) -> Vec<Dd> {
    let mut x = b.to_vec();
    for i in 0..n {
        for k in 0..i {
            x[i] = x[i] - l[i * n + k] * x[k];
        }
        x[i] = x[i] / l[i * n + i];
    }
    for i in (0..n).rev() {
        for k in i + 1..n {
            x[i] = x[i] - l[k * n + i] * x[k];
        }
        x[i] = x[i] / l[i * n + i];
    }
    x
}

/// `ln |L Lᵀ|` of a factor `l`.
fn log_det(l: &[Dd], n: usize) -> f64 {
    2.0 * (0..n).map(|i| l[i * n + i].ln_f64()).sum::<f64>()
}

/// RBF kernel with signal variance 1 and lengthscales `ls` (one per
/// feature, or one for all) between row `i` of the column-major `a`
/// (`na × d`) and row `j` of `b` (`nb × d`).
fn rbf(a: &[f64], na: usize, i: usize, b: &[f64], nb: usize, j: usize, ls: &[f64]) -> Dd {
    let d = a.len() / na;
    let mut s = Dd::ZERO;
    for f in 0..d {
        let diff = Dd::new(a[f * na + i]) - Dd::new(b[f * nb + j]);
        let l = ls[f % ls.len()];
        s = s + diff * diff / (Dd::new(l) * Dd::new(l));
    }
    (-(s * Dd::new(0.5))).exp()
}

/// `a` (column-major `n × d`) without the rows that repeat an earlier row
/// exactly: a VFE system with a repeated inducing point is the system
/// without the repeat.
fn without_repeats(a: &[f64], n: usize, d: usize) -> (Vec<f64>, usize) {
    let rows: Vec<usize> = (0..n)
        .filter(|&i| {
            !(0..i).any(|k| (0..d).all(|f| a[f * n + i].to_bits() == a[f * n + k].to_bits()))
        })
        .collect();
    let out = (0..d)
        .flat_map(|f| rows.iter().map(move |&i| a[f * n + i]))
        .collect();
    (out, rows.len())
}

/// The mean and the variance (with the noise) at `xs` and the NLML of the
/// Sparse VFE system on `x` (`n × d`, column-major), `y`, and the inducing
/// points `z` (`m × d`), for the RBF kernel with lengthscales `ls` and
/// noise variance `noise`, in the order of `Prediction` then the NLML:
/// `q` means, `q` variances, one NLML. Computed in double-double from
/// `Σ = σ² K_mm + K_mx K_xm`, so `K_mm` is never factored on its own
/// where a conditioning of `1e15` would cost the values.
pub fn vfe_reference(
    ls: &[f64],
    noise: f64,
    x: &[f64],
    y: &[f64],
    z: &[f64],
    xs: &[f64],
    d: usize,
) -> Vec<f64> {
    let n = y.len();
    let (z, m) = without_repeats(z, z.len() / d, d);
    let q = xs.len() / d;
    let s2 = Dd::new(noise);
    let mut k_mm = vec![Dd::ZERO; m * m];
    for i in 0..m {
        for j in 0..m {
            k_mm[i * m + j] = rbf(&z, m, i, &z, m, j, ls);
        }
    }
    let mut k_mx = vec![Dd::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            k_mx[i * n + j] = rbf(&z, m, i, x, n, j, ls);
        }
    }
    // `Σ = σ² K_mm + K_mx K_xm` and `K_mx K_xm`.
    let mut gram = vec![Dd::ZERO; m * m];
    for i in 0..m {
        for j in 0..m {
            let mut v = Dd::ZERO;
            for k in 0..n {
                v = v + k_mx[i * n + k] * k_mx[j * n + k];
            }
            gram[i * m + j] = v;
        }
    }
    let sigma: Vec<Dd> = (0..m * m).map(|i| s2 * k_mm[i] + gram[i]).collect();
    let (Some(l_mm), Some(l_sigma)) = (cholesky(&k_mm, m), cholesky(&sigma, m)) else {
        return vec![f64::NAN; 2 * q + 1];
    };
    let k_y: Vec<Dd> = (0..m)
        .map(|i| (0..n).fold(Dd::ZERO, |v, k| v + k_mx[i * n + k] * Dd::new(y[k])))
        .collect();
    let sigma_inv_ky = solve(&l_sigma, m, &k_y);
    let mut out = Vec::with_capacity(2 * q + 1);
    let k_sm: Vec<Vec<Dd>> = (0..q)
        .map(|r| (0..m).map(|j| rbf(xs, q, r, &z, m, j, ls)).collect())
        .collect();
    for row in &k_sm {
        let mean = (0..m).fold(Dd::ZERO, |v, j| v + row[j] * sigma_inv_ky[j]);
        out.push(mean.to_f64());
    }
    for row in &k_sm {
        let a = solve(&l_mm, m, row);
        let b = solve(&l_sigma, m, row);
        let mut v = Dd::ONE + s2;
        for j in 0..m {
            v = v - row[j] * a[j] + s2 * row[j] * b[j];
        }
        out.push(v.to_f64());
    }
    let y_norm2 = y.iter().fold(Dd::ZERO, |v, &t| v + Dd::new(t) * Dd::new(t));
    let ky_s_ky = (0..m).fold(Dd::ZERO, |v, i| v + k_y[i] * sigma_inv_ky[i]);
    let quad = (y_norm2 - ky_s_ky) / s2;
    let mut trace = Dd::ZERO;
    for j in 0..m {
        let col: Vec<Dd> = (0..m).map(|i| gram[i * m + j]).collect();
        trace = trace + solve(&l_mm, m, &col)[j];
    }
    let two_pi_ln = (2.0 * std::f64::consts::PI).ln();
    let nlml = 0.5
        * (n as f64 * two_pi_ln + (n - m) as f64 * noise.ln() + log_det(&l_sigma, m)
            - log_det(&l_mm, m)
            + quad.to_f64())
        + (Dd::new(n as f64) - trace).to_f64() / (2.0 * noise);
    out.push(nlml);
    out
}
