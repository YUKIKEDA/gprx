//! Seeded random numbers for sampling, restarts, FSA, and the Adam shuffle.
//!
//! The generator is Xoshiro256++ (Blackman and Vigna) with its state filled
//! from the seed by SplitMix64, written here so the sequence for a seed is
//! fixed by gprx alone: the same seed gives the same numbers on every
//! platform and with any version of the `rand` crate. Changing that sequence
//! is a breaking change of gprx.

/// Xoshiro256++ state. Crate-private: the public surface takes a `u64` seed.
#[derive(Clone, Debug)]
pub(crate) struct SeededRng {
    s: [u64; 4],
}

impl SeededRng {
    /// The state SplitMix64 produces from `seed`, as in the reference
    /// `seed_from_u64` of `rand_xoshiro`.
    pub(crate) fn new(seed: u64) -> Self {
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
        }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)` from the top 53 bits.
    pub(crate) fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `0..=max`, without modulo bias (rejection on the widened
    /// product, Lemire 2019).
    pub(crate) fn up_to(&mut self, max: usize) -> usize {
        let range = max as u64 + 1;
        if range == 0 {
            return self.next_u64() as usize;
        }
        let threshold = range.wrapping_neg() % range;
        loop {
            let wide = u128::from(self.next_u64()) * u128::from(range);
            if (wide as u64) >= threshold {
                return (wide >> 64) as usize;
            }
        }
    }
}

pub(crate) fn seeded_rng(seed: u64) -> SeededRng {
    SeededRng::new(seed)
}

pub(crate) fn open_unit(rng: &mut SeededRng) -> f64 {
    let u = rng.unit();
    let eps = 1.0 / ((1u64 << 53) as f64);
    if u <= eps {
        eps
    } else if u >= 1.0 - eps {
        1.0 - eps
    } else {
        u
    }
}

pub(crate) fn unit_normal(rng: &mut SeededRng) -> f64 {
    let u1 = open_unit(rng).max(f64::MIN_POSITIVE);
    let u2 = open_unit(rng);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {

    /// The first outputs for seeds 0 and 42, checked against
    /// `rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64` 0.7. A change here
    /// changes every seeded result of gprx.
    #[test]
    fn sequence_is_pinned() {
        let mut rng = super::SeededRng::new(0);
        let got: Vec<u64> = (0..4).map(|_| rng.next_u64()).collect();
        assert_eq!(got, SEED_0);
        let mut rng = super::SeededRng::new(42);
        let got: Vec<u64> = (0..4).map(|_| rng.next_u64()).collect();
        assert_eq!(got, SEED_42);
    }

    const SEED_0: [u64; 4] = [
        0x5317_5d61_490b_23df,
        0x61da_6f3d_c380_d507,
        0x5c0f_df91_ec9a_7bfc,
        0x02ee_bf8c_3bbe_5e1a,
    ];
    const SEED_42: [u64; 4] = [
        0xd076_4d4f_4476_689f,
        0x519e_4174_576f_3791,
        0xfbe0_7cfb_0c24_ed8c,
        0xb37d_9f60_0cd8_35b8,
    ];

    #[test]
    fn up_to_stays_in_range_and_unit_in_half_open_interval() {
        let mut rng = super::SeededRng::new(7);
        for max in [0usize, 1, 2, 9, 1000] {
            for _ in 0..200 {
                assert!(rng.up_to(max) <= max);
            }
        }
        for _ in 0..1000 {
            let u = rng.unit();
            assert!((0.0..1.0).contains(&u));
        }
    }
}
