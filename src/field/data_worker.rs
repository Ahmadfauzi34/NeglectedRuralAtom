/// State of a Data Worker.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
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

    // Abstract payloads: URLs, File Paths, or raw strings.
    // In a highly optimized setup, this would be a single massive String
    // with index slices (start, end). For simplicity in script interop, we use Vec<String>.
    pub(crate) payloads: Vec<String>,
    pub(crate) results: Vec<String>,

    pub(crate) len: usize,
    capacity: usize,
}

impl DataWorkerField {
    pub fn new(capacity: usize) -> Self {
        Self {
            active: Vec::with_capacity(capacity),
            states: Vec::with_capacity(capacity),
            task_ids: Vec::with_capacity(capacity),
            progress: Vec::with_capacity(capacity),
            payloads: Vec::with_capacity(capacity),
            results: Vec::with_capacity(capacity),
            len: 0,
            capacity,
        }
    }

    pub fn spawn_worker(&mut self, task_id: u32, payload: &str) -> usize {
        let idx = self.len;

        self.active.push(1);
        self.states.push(WorkerState::Idle as u8);
        self.task_ids.push(task_id);
        self.progress.push(0.0);
        self.payloads.push(payload.to_string());
        self.results.push(String::new());

        self.len += 1;
        idx
    }

    pub fn kill_swap(&mut self, idx: usize) {
        if idx >= self.len { return; }

        let last = self.len - 1;
        if idx != last {
            self.active.swap(idx, last);
            self.states.swap(idx, last);
            self.task_ids.swap(idx, last);
            self.progress.swap(idx, last);
            self.payloads.swap(idx, last);
            self.results.swap(idx, last);
        }

        // Logical remove (data remains in vector but won't be iterated)
        self.len -= 1;
    }
}
