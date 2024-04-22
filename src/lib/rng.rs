use std::f64::consts::PI;

/// quick and dirty RNG
pub struct Rng {
    state: u64,
}

impl Rng {
    /// seed must be non-zero
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn rand(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;

        self.state
    }
}

/// function for sampling from a normal distribution using a random number
/// generator that generates a uniform distribution
pub fn box_muller(rng: &mut Rng, mean: f64, std_dev: f64) -> f64 {
    let u1 = (rng.rand() as f64) / (std::u64::MAX as f64);
    let u2 = (rng.rand() as f64) / (std::u64::MAX as f64);

    let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();

    z1 * std_dev + mean
}
