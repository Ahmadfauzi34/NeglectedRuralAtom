use std::fmt::Write;

/// A highly optimized module to convert internal agent states into textual
/// descriptions (prompts) for LLM consumption.
pub struct PromptBuilder {
    // Pre-allocated buffer to prevent multiple heap allocations during prompt building
    buffer: String,
}

pub struct SystemSnapshot {
    pub agents: Vec<(usize, f32, f32, f32, f32, f32)>,
    pub workers: Vec<(usize, u32, u8, f32)>,
    pub vfs_files: Vec<String>,
    pub metrics: crate::telemetry::EngineMetrics,
    pub total_agents: usize,
}

impl PromptBuilder {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: String::with_capacity(capacity),
        }
    }

    /// Iterates over a fast snapshot of the system to build a comprehensive data-centric prompt.
    /// This avoids visual bias by reporting on workers, storage, and performance metrics.
    pub fn build_from_snapshot(&mut self, snap: &SystemSnapshot) -> &str {
        self.buffer.clear();

        let _ = writeln!(&mut self.buffer, "--- KERNEL DATA STATUS ---");

        // 1. Performance & Load Metrics
        let _ = writeln!(
            &mut self.buffer,
            "Metrics: physics_ms={:.2}, script_ms={:.2}, workers_active={}, arena_bytes={}",
            snap.metrics.physics_step_ms,
            snap.metrics.scripting_eval_ms,
            snap.metrics.active_data_workers,
            snap.metrics.text_arena_bytes
        );

        // 2. Data Workers (The core of data processing)
        if !snap.workers.is_empty() {
            let _ = writeln!(&mut self.buffer, "\nActive Data Workers:");
            for &(idx, task_id, state, progress) in &snap.workers {
                let state_str = match state {
                    1 => "Working",
                    2 => "Done",
                    3 => "Error",
                    _ => "Idle",
                };
                let _ = writeln!(
                    &mut self.buffer,
                    "- Worker {idx}: task={task_id}, state={state_str}, progress={progress:.1}%"
                );
            }
        }

        // 3. Virtual File System (Knowledge & Memory)
        if !snap.vfs_files.is_empty() {
            let _ = writeln!(&mut self.buffer, "\nVFS Storage:");
            for path in &snap.vfs_files {
                let _ = writeln!(&mut self.buffer, "- {path}");
            }
        }

        // 4. Visual/Physics Agents (Optional context)
        if !snap.agents.is_empty() {
            let _ = writeln!(&mut self.buffer, "\nVisual Agent Context:");
            for &(id, px, py, _, _, health) in &snap.agents {
                let _ = writeln!(
                    &mut self.buffer,
                    "- ID {id}: pos({px:.0},{py:.0}), hp:{health:.1}"
                );
            }
            if snap.total_agents > snap.agents.len() {
                let _ = writeln!(
                    &mut self.buffer,
                    "... and {} more agents.",
                    snap.total_agents - snap.agents.len()
                );
            }
        }

        &self.buffer
    }
}
