//! Seeded [`rand::rngs::SmallRng`] shared by sampling, restarts, FSA, and Adam.

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

pub(crate) fn small_rng(seed: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed)
}

pub(crate) fn open_unit(rng: &mut SmallRng) -> f64 {
    let u: f64 = rng.random();
    let eps = 1.0 / ((1u64 << 53) as f64);
    if u <= eps {
        eps
    } else if u >= 1.0 - eps {
        1.0 - eps
    } else {
        u
    }
}

pub(crate) fn unit_normal(rng: &mut SmallRng) -> f64 {
    let u1 = open_unit(rng).max(f64::MIN_POSITIVE);
    let u2 = open_unit(rng);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}
