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

    // Cached inverse magnitudes to speed up cosine similarity
    pub(crate) inv_magnitudes: Vec<f32>,

    pub(crate) len: usize,

    // Capacity limit to prevent unbounded memory growth
    pub(crate) max_capacity: usize,

    // Dynamic tracking of total bytes allocated by memory_ids
    pub(crate) total_id_bytes: usize,
    pub(crate) max_id_bytes: usize,
}

impl VectorMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            memory_ids: Vec::with_capacity(capacity),
            vectors: Vec::with_capacity(capacity * EMBEDDING_DIM),
            inv_magnitudes: Vec::with_capacity(capacity),
            len: 0,
            max_capacity: capacity,
            total_id_bytes: 0,
            // Dynamic capacity pool (e.g. max_capacity * 1024 bytes) allows large textual payloads
            // as long as the global quota is not reached.
            max_id_bytes: capacity * 1024,
        }
    }

    /// Stores a new vector embedding into memory.
    pub fn store(&mut self, memory_id: &str, vector: &[f32]) {
        if vector.len() != EMBEDDING_DIM {
            return; // Ignore malformed vectors
        }

        // Anti-memory-leak: Prevent unbounded growth from malicious/runaway LLM scripts
        if self.len >= self.max_capacity {
            return;
        }

        let mut safe_id = memory_id.to_string();
        let projected_bytes = self.total_id_bytes + safe_id.len();

        // Dynamic capacity fallback instead of rigid 1024-byte hard limit
        if projected_bytes > self.max_id_bytes {
            let available = self.max_id_bytes.saturating_sub(self.total_id_bytes);
            if available == 0 {
                return; // Drops entirely if no text quota remains
            }
            let mut end = available;
            if end > safe_id.len() {
                end = safe_id.len();
            }
            while end > 0 && !safe_id.is_char_boundary(end) {
                end -= 1;
            }
            safe_id.truncate(end);
        }

        self.total_id_bytes += safe_id.len();
        self.memory_ids.push(safe_id);
        self.vectors.extend_from_slice(vector);

        let mag_sq: f32 = vector.iter().map(|v| v * v).sum();
        // Optimize: Store the inverse magnitude so search() only needs to multiply
        self.inv_magnitudes.push(1.0 / (mag_sq.sqrt() + 1e-6));

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

        // Calculate query inverse magnitude for cosine similarity
        let query_mag_sq: f32 = query_vector.iter().map(|v| v * v).sum();
        let inv_query_mag = 1.0 / (query_mag_sq.sqrt() + 1e-6);

        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for i in 0..self.len {
            let start = i * EMBEDDING_DIM;
            let target_vec = &self.vectors[start..start + EMBEDDING_DIM];

            let dot = self.dot_product(query_vector, target_vec);
            let inv_target_mag = self.inv_magnitudes[i];

            // Optimized Cosine Similarity: O(1) multiplications instead of divisions
            let cosine_similarity = dot * inv_query_mag * inv_target_mag;

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
        self.inv_magnitudes.clear();
        self.len = 0;
        self.total_id_bytes = 0;
    }
}
