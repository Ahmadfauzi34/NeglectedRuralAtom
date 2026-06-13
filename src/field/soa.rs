use wasm_bindgen::prelude::*;

/// SOA: Structure of Arrays — cache-friendly, SIMD-ready, no indirection
///
/// Prinsip:
/// - Semua field hot path dalam Vec terpisah (continuous memory)
/// - Pre-allocated capacity, tidak alloc di hot loop
/// - Ghost state: 'len' tracking, data "mati" tetap di buffer tapi di-swap ke belakang
#[wasm_bindgen]
pub struct AgentField {
    // Hot path fields — continuous, aligned
    pub(crate) pos_x: Vec<f32>,
    pub(crate) pos_y: Vec<f32>,
    pub(crate) vel_x: Vec<f32>,
    pub(crate) vel_y: Vec<f32>,

    // Cold fields — bisa dipisah ke buffer terpisah kalau perlu
    pub(crate) health: Vec<f32>,
    pub(crate) target_id: Vec<u32>,
    pub(crate) active: Vec<u8>, // 0 = mati (ghost), 1 = aktif

    // Behavior / State Machine memory
    // 0 = Idle, 1 = Wandering, 2 = Chasing, 3 = Fleeing
    pub(crate) behavior_state: Vec<u8>,

    // Pre-allocated scratch buffers for physics iteration to enforce Zero-Allocation in hot loop
    pub(crate) acc_x: Vec<f32>,
    pub(crate) acc_y: Vec<f32>,

    // Ghost state tracking
    pub(crate) len: usize,
    pub(crate) capacity: usize,
}

#[wasm_bindgen]
impl AgentField {
    #[wasm_bindgen(constructor)]
    pub fn new(initial_capacity: usize) -> Self {
        Self {
            pos_x: Vec::with_capacity(initial_capacity),
            pos_y: Vec::with_capacity(initial_capacity),
            vel_x: Vec::with_capacity(initial_capacity),
            vel_y: Vec::with_capacity(initial_capacity),
            health: Vec::with_capacity(initial_capacity),
            target_id: Vec::with_capacity(initial_capacity),
            active: Vec::with_capacity(initial_capacity),
            behavior_state: Vec::with_capacity(initial_capacity),
            acc_x: Vec::with_capacity(initial_capacity),
            acc_y: Vec::with_capacity(initial_capacity),
            len: 0,
            capacity: initial_capacity,
        }
    }

    /// Pre-grow buffer untuk mencegah alloc di runtime
    pub fn reserve(&mut self, additional: usize) {
        let new_cap = self.len + additional;
        self.pos_x
            .reserve(new_cap.saturating_sub(self.pos_x.capacity()));
        self.pos_y
            .reserve(new_cap.saturating_sub(self.pos_y.capacity()));
        self.vel_x
            .reserve(new_cap.saturating_sub(self.vel_x.capacity()));
        self.vel_y
            .reserve(new_cap.saturating_sub(self.vel_y.capacity()));
        self.health
            .reserve(new_cap.saturating_sub(self.health.capacity()));
        self.target_id
            .reserve(new_cap.saturating_sub(self.target_id.capacity()));
        self.active
            .reserve(new_cap.saturating_sub(self.active.capacity()));
        self.behavior_state
            .reserve(new_cap.saturating_sub(self.behavior_state.capacity()));
        self.acc_x
            .reserve(new_cap.saturating_sub(self.acc_x.capacity()));
        self.acc_y
            .reserve(new_cap.saturating_sub(self.acc_y.capacity()));
        self.capacity = self.pos_x.capacity();
    }

    /// Spawn agent — hanya push ke back, tidak alloc kalau capacity cukup
    pub fn spawn(&mut self, x: f32, y: f32, health: f32) -> usize {
        let idx = self.len;

        // Pastikan capacity cukup — ini jarang terjadi kalau reserve() dipanggil di init
        if idx >= self.capacity {
            self.reserve(self.capacity.max(64));
        }

        self.pos_x.push(x);
        self.pos_y.push(y);
        self.vel_x.push(0.0);
        self.vel_y.push(0.0);
        self.health.push(health);
        self.target_id.push(u32::MAX); // none
        self.active.push(1);
        self.behavior_state.push(0); // Idle default
        self.acc_x.push(0.0);
        self.acc_y.push(0.0);

        self.len += 1;
        idx
    }

    /// Ghost state removal: swap-drop alih-alih `Vec::remove`
    /// O(1), tidak shifting, buffer tidak di-free
    pub fn kill_swap(&mut self, idx: usize) {
        if idx >= self.len {
            return;
        }

        let last = self.len - 1;
        if idx != last {
            self.pos_x.swap(idx, last);
            self.pos_y.swap(idx, last);
            self.vel_x.swap(idx, last);
            self.vel_y.swap(idx, last);
            self.health.swap(idx, last);
            self.target_id.swap(idx, last);
            self.active.swap(idx, last);
            self.behavior_state.swap(idx, last);
            self.acc_x.swap(idx, last);
            self.acc_y.swap(idx, last);
        }

        // Ghost: data di 'last' sekarang invalid, tapi buffer tetap ada
        self.len -= 1;
    }

    // === Direct pointer access untuk zero-copy ke JS/Canvas ===

    pub fn pos_x_ptr(&self) -> *const f32 {
        self.pos_x.as_ptr()
    }

    pub fn pos_y_ptr(&self) -> *const f32 {
        self.pos_y.as_ptr()
    }

    pub fn vel_x_ptr(&self) -> *const f32 {
        self.vel_x.as_ptr()
    }

    pub fn vel_y_ptr(&self) -> *const f32 {
        self.vel_y.as_ptr()
    }

    pub fn active_ptr(&self) -> *const u8 {
        self.active.as_ptr()
    }

    pub fn agent_count(&self) -> usize {
        self.len
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}
