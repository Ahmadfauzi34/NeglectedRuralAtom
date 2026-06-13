use std::fmt::Write;

/// A highly optimized module to convert internal agent states into textual
/// descriptions (prompts) for LLM consumption.
pub struct PromptBuilder {
    // Pre-allocated buffer to prevent multiple heap allocations during prompt building
    buffer: String,
}

impl PromptBuilder {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: String::with_capacity(capacity),
        }
    }

    /// Iterates over a fast snapshot of agents to avoid holding read locks.
    /// This is highly optimized using `Write` trait to avoid `format!` allocations.
    pub fn build_from_snapshot(
        &mut self,
        snapshot: &[(usize, f32, f32, f32, f32, f32)],
        total_agents: usize,
    ) -> &str {
        self.buffer.clear();

        let _ = writeln!(&mut self.buffer, "Current Agent States:");

        for &(id, px, py, vx, vy, health) in snapshot {
            let _ = writeln!(
                &mut self.buffer,
                "Agent ID {id}: pos(x:{px:.1}, y:{py:.1}), vel(vx:{vx:.2}, vy:{vy:.2}), health: {health:.1}"
            );
        }

        if total_agents > snapshot.len() {
            let _ = writeln!(
                &mut self.buffer,
                "... and {} more agents.",
                total_agents - snapshot.len()
            );
        }

        &self.buffer
    }
}
