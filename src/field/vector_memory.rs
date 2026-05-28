/// Constant defining the dimensionality of the vector embeddings.
/// Fixed to 16 to maintain cache locality and zero-allocation speeds in WASM.
pub const EMBEDDING_DIM: usize = 16;

/// A lightweight, high-performance Structure of Arrays (SOA) Vector Database.
/// Designed for client-side Retrieval-Augmented Generation (RAG) and Agentic Micro-Memory.
pub struct VectorMemory {
    // Stores strings representing the memory "ID" or "Payload Data"
    pub(crate) memory_ids: Vec<String>,

    // A flattened 1D array representing a 2D matrix of embeddings (count * EMBEDDING_DIM)
    pub(crate) vectors: Vec<f32>,

    pub(crate) len: usize,
}

impl VectorMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            memory_ids: Vec::with_capacity(capacity),
            vectors: Vec::with_capacity(capacity * EMBEDDING_DIM),
            len: 0,
        }
    }

    /// Stores a new vector embedding into memory.
    pub fn store(&mut self, memory_id: &str, vector: &[f32]) {
        if vector.len() != EMBEDDING_DIM {
            return; // Ignore malformed vectors
        }

        self.memory_ids.push(memory_id.to_string());
        self.vectors.extend_from_slice(vector);
        self.len += 1;
    }

    /// Calculates the dot product of two vectors (assumes vectors are already normalized).
    #[inline(always)]
    fn dot_product(&self, vec_a: &[f32], vec_b: &[f32]) -> f32 {
        let mut sum = 0.0;
        // In a production engine, this loop would automatically be auto-vectorized
        // via SIMD by LLVM if target-feature=+simd128 is enabled.
        for i in 0..EMBEDDING_DIM {
            sum += vec_a[i] * vec_b[i];
        }
        sum
    }

    /// Searches the entire memory bank for the highest cosine similarity match.
    /// Returns the nearest memory_id and its similarity score.
    pub fn search(&self, query_vector: &[f32]) -> Option<(String, f32)> {
        if self.len == 0 || query_vector.len() != EMBEDDING_DIM {
            return None;
        }

        // Calculate query magnitude for cosine similarity
        let query_mag_sq: f32 = query_vector.iter().map(|v| v * v).sum();
        let query_mag = query_mag_sq.sqrt() + 1e-6;

        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for i in 0..self.len {
            let start = i * EMBEDDING_DIM;
            let target_vec = &self.vectors[start..start + EMBEDDING_DIM];

            let dot = self.dot_product(query_vector, target_vec);
            let target_mag_sq: f32 = target_vec.iter().map(|v| v * v).sum();
            let target_mag = target_mag_sq.sqrt() + 1e-6;

            let cosine_similarity = dot / (query_mag * target_mag);

            if cosine_similarity > best_score {
                best_score = cosine_similarity;
                best_idx = i;
            }
        }

        Some((self.memory_ids[best_idx].clone(), best_score))
    }

    pub fn clear(&mut self) {
        self.memory_ids.clear();
        self.vectors.clear();
        self.len = 0;
    }
}
