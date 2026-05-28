use serde::Deserialize;
use crate::field::{AgentField, KernelConfig};

#[derive(Debug, Deserialize)]
#[serde(tag = "cmd")]
pub enum Command {
    #[serde(rename = "spawn")]
    Spawn { x: f32, y: f32, vx: f32, vy: f32, health: f32, #[allow(dead_code)] color: u32 },
    
    #[serde(rename = "kill")]
    Kill { idx: usize },
    
    #[serde(rename = "config")]
    Config {
        dt: f32, friction: f32, max_speed: f32, influence_radius: f32,
        #[serde(default)] cursor_x: f32,
        #[serde(default)] cursor_y: f32,
        #[serde(default)] cursor_weight: f32
    },
    
    #[serde(rename = "clear")]
    Clear,
    
    #[serde(rename = "batch")]
    Batch(Vec<Command>),  // <-- FIX: hapus <
}

pub struct CommandBus {
    queue: Vec<Command>,  // <-- FIX: hapus <
}

impl CommandBus {
    pub fn new() -> Self {
        Self { queue: Vec::with_capacity(64) }
    }
    
    pub fn parse(&mut self, json: &str) -> Result<(), serde_json::Error> {
        let cmd: Command = serde_json::from_str(json)?;
        self.queue.push(cmd);
        Ok(())
    }
    
    pub fn parse_batch(&mut self, json: &str) -> Result<(), serde_json::Error> {
        let cmds: Vec<Command> = serde_json::from_str(json)?;  // <-- FIX: hapus <
        self.queue.extend(cmds);
        Ok(())
    }
    
    pub fn execute(&mut self, field: &mut AgentField, config: &mut KernelConfig) {
        // Drain + iterasi — hindari alloc
        // We need to collect the drained commands first or process batch commands differently
        // to avoid mutable borrow alias
        let cmds = std::mem::take(&mut self.queue);
        for cmd in cmds {
            match cmd {
                Command::Spawn { x, y, vx, vy, health, color: _ } => {
                    let idx = field.spawn(x, y, health);
                    if idx < field.vel_x.len() {
                        field.vel_x[idx] = vx;
                        field.vel_y[idx] = vy;
                    }
                }
                Command::Kill { idx } => field.kill_swap(idx),
                Command::Config { dt, friction, max_speed, influence_radius, cursor_x, cursor_y, cursor_weight } => {
                    *config = KernelConfig {
                        dt,
                        friction,
                        max_speed,
                        influence_radius,
                        cursor_x,
                        cursor_y,
                        cursor_weight,
                        ..*config
                    };
                }
                Command::Clear => {
                    field.len = 0;
                }
                Command::Batch(batch) => {
                    // Flatten: extend queue, dieksekusi di drain berikutnya
                    // Tapi karena kita sudah dalam drain, masukkan ke queue baru
                    self.queue.extend(batch);
                }
            }
        }
    }
    
    pub fn clear(&mut self) {
        self.queue.clear();
    }
}
