use crate::field::AgentField;
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

    /// Iterates over active agents and builds a concise text representation.
    /// This is highly optimized using `Write` trait to avoid `format!` allocations.
    pub fn build_agent_state_prompt(&mut self, field: &AgentField) -> &str {
        self.buffer.clear();

        let _ = writeln!(&mut self.buffer, "Current Agent States:");

        // Only summarize the first N agents to avoid exceeding LLM context limits
        let limit = field.len.min(50);
        let mut count = 0;

        for i in 0..field.len {
            if count >= limit {
                break;
            }
            if field.active[i] == 1 {
                let _ = writeln!(
                    &mut self.buffer,
                    "Agent ID {}: pos(x:{:.1}, y:{:.1}), vel(vx:{:.2}, vy:{:.2}), health: {:.1}",
                    i, field.pos_x[i], field.pos_y[i], field.vel_x[i], field.vel_y[i], field.health[i]
                );
                count += 1;
            }
        }

        if field.len > limit {
            let _ = writeln!(&mut self.buffer, "... and {} more agents.", field.len - limit);
        }

        &self.buffer
    }
}
