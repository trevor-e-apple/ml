pub fn add(left: usize, right: usize) -> usize {
    left + right
}

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
