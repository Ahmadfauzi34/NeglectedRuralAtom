/// State of a Data Worker.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum WorkerState {
    Idle = 0,
    Working = 1,
    Done = 2,
    Error = 3,
}

/// SOA (Structure of Arrays) mapping for non-visual data agents.
/// Designed for tasks like code analysis, web crawling, or prompt distribution.
pub struct DataWorkerField {
    pub(crate) active: Vec<u8>,
    pub(crate) states: Vec<u8>,
    pub(crate) task_ids: Vec<u32>,
    pub(crate) progress: Vec<f32>,

    // Autonomy: Local Memory for Q-Learning or Neural Net Weights
    // Each worker gets an 8-float array for extremely fast WASM native math
    pub(crate) memory: Vec<[f32; 8]>,

    pub(crate) text_arena: String,
    pub(crate) payload_slices: Vec<(u32, u32)>,
    pub(crate) result_slices: Vec<(u32, u32)>,

    pub(crate) free_slots: Vec<usize>,
    pub(crate) len: usize,
    capacity: usize,
}

impl DataWorkerField {
    pub fn new(capacity: usize) -> Self {
        // Initialize free_slots in reverse order (LIFO stack pop from end)
        let mut free_slots = Vec::with_capacity(capacity);
        for i in (0..capacity).rev() {
            free_slots.push(i);
        }

        Self {
            active: vec![0; capacity],
            states: vec![0; capacity],
            task_ids: vec![0; capacity],
            progress: vec![0.0; capacity],
            memory: vec![[0.0; 8]; capacity],
            text_arena: String::with_capacity(capacity * 256),
            payload_slices: vec![(0, 0); capacity],
            result_slices: vec![(0, 0); capacity],
            free_slots,
            len: 0,
            capacity,
        }
    }

    /// Spawn worker O(1) using Free-List Stack
    pub fn spawn_worker(&mut self, task_id: u32, payload: &str) -> i32 {
        let idx = match self.free_slots.pop() {
            Some(i) => i,
            None => return -1, // Buffer full
        };

        let p_start = self.text_arena.len() as u32;
        self.text_arena.push_str(payload);
        let p_end = self.text_arena.len() as u32;

        self.active[idx] = 1;
        self.states[idx] = WorkerState::Working as u8;
        self.task_ids[idx] = task_id;
        self.progress[idx] = 0.0;
        self.payload_slices[idx] = (p_start, p_end);
        self.result_slices[idx] = (0, 0);

        // Track highest logical index for loops
        if idx >= self.len {
            self.len = idx + 1;
        }

        idx as i32
    }

    pub fn kill_worker(&mut self, idx: usize) {
        if idx >= self.capacity || self.active[idx] == 0 { return; }

        self.active[idx] = 0;
        self.states[idx] = WorkerState::Idle as u8;

        // Return index to stack to be reused
        self.free_slots.push(idx);
    }

    /// Manual Garbage Collection
    /// Compacts the massive string arena, dropping text from dead agents
    pub fn compact_arena(&mut self) {
        let mut new_arena = String::with_capacity(self.text_arena.len());

        for i in 0..self.len {
            if self.active[i] == 1 {
                // Relocate Payload
                let (p_start, p_end) = self.payload_slices[i];
                if p_start != p_end {
                    if let Some(text) = self.text_arena.get(p_start as usize..p_end as usize) {
                        let new_start = new_arena.len() as u32;
                        new_arena.push_str(text);
                        self.payload_slices[i] = (new_start, new_arena.len() as u32);
                    } else {
                        self.payload_slices[i] = (new_arena.len() as u32, new_arena.len() as u32);
                    }
                } else {
                    self.payload_slices[i] = (new_arena.len() as u32, new_arena.len() as u32);
                }

                // Relocate Result
                let (r_start, r_end) = self.result_slices[i];
                if r_start != r_end {
                    if let Some(text) = self.text_arena.get(r_start as usize..r_end as usize) {
                        let new_start = new_arena.len() as u32;
                        new_arena.push_str(text);
                        self.result_slices[i] = (new_start, new_arena.len() as u32);
                    } else {
                        self.result_slices[i] = (new_arena.len() as u32, new_arena.len() as u32);
                    }
                } else {
                    self.result_slices[i] = (new_arena.len() as u32, new_arena.len() as u32);
                }
            } else {
                // Reset dead slices
                self.payload_slices[i] = (0, 0);
                self.result_slices[i] = (0, 0);
            }
        }
        self.text_arena = new_arena;
    }
}
