//! Deterministic synthetic dataset generation.
//!
//! No `rand` dependency — a tiny splitmix64 PRNG gives reproducible vectors from a
//! seed, so every engine sees byte-identical input and runs are comparable over time.

/// splitmix64 — a fast, well-distributed seedable PRNG (public-domain algorithm).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f32` in `[-1.0, 1.0)`.
    #[inline]
    fn next_f32(&mut self) -> f32 {
        // Top 24 bits -> [0, 1), then map to [-1, 1).
        let unit = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        unit * 2.0 - 1.0
    }
}

/// A generated dataset: `n` corpus vectors and `m` query vectors, both row-major
/// flat `f32` buffers of stride `dim`. Ids are simply `0..n`.
pub struct Dataset {
    pub dim: usize,
    pub ids: Vec<u64>,
    pub vectors: Vec<f32>,
    pub queries: Vec<Vec<f32>>,
}

impl Dataset {
    pub fn n(&self) -> usize {
        self.ids.len()
    }
}

/// Generate a dataset. Corpus and queries use distinct sub-seeds so a query is not
/// trivially identical to a stored vector.
pub fn generate(seed: u64, n: usize, dim: usize, num_queries: usize) -> Dataset {
    let mut corpus_rng = Rng::new(seed);
    let mut vectors = Vec::with_capacity(n * dim);
    for _ in 0..n * dim {
        vectors.push(corpus_rng.next_f32());
    }

    let mut query_rng = Rng::new(seed ^ 0xD1B5_4A32_D192_ED03);
    let mut queries = Vec::with_capacity(num_queries);
    for _ in 0..num_queries {
        let mut q = Vec::with_capacity(dim);
        for _ in 0..dim {
            q.push(query_rng.next_f32());
        }
        queries.push(q);
    }

    Dataset {
        dim,
        ids: (0..n as u64).collect(),
        vectors,
        queries,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_across_runs() {
        let a = generate(7, 100, 16, 5);
        let b = generate(7, 100, 16, 5);
        assert_eq!(a.vectors, b.vectors);
        assert_eq!(a.queries, b.queries);
    }

    #[test]
    fn shapes_are_correct() {
        let d = generate(1, 50, 8, 4);
        assert_eq!(d.vectors.len(), 50 * 8);
        assert_eq!(d.ids.len(), 50);
        assert_eq!(d.queries.len(), 4);
        assert!(d.queries.iter().all(|q| q.len() == 8));
    }

    #[test]
    fn values_in_range() {
        let d = generate(99, 1000, 4, 0);
        assert!(d.vectors.iter().all(|&x| (-1.0..1.0).contains(&x)));
    }
}
