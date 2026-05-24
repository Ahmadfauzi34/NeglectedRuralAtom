use serde::Deserialize;
use crate::field::{AgentField, KernelConfig};

#[derive(Debug, Deserialize)]
#[serde(tag = "cmd")]
pub enum Command {
    #[serde(rename = "spawn")]
    Spawn { x: f32, y: f32, vx: f32, vy: f32, health: f32, color: u32 },
    
    #[serde(rename = "kill")]
    Kill { idx: usize },
    
    #[serde(rename = "config")]
    Config { dt: f32, friction: f32, max_speed: f32, influence_radius: f32 },
    
    #[serde(rename = "clear")]
    Clear,
    
    #[serde(rename = "batch")]
    Batch(Vec<<Command>),
}

pub struct CommandBus {
    queue: Vec<<Command>,
}

impl CommandBus {
    pub fn new() -> Self {
        Self { queue: Vec::with_capacity(64) }
    }
    
    /// Parse single command dari JSON string. Dipanggil dari JS.
    pub fn parse(&mut self, json: &str) -> Result<(), serde_json::Error> {
        let cmd: Command = serde_json::from_str(json)?;
        self.queue.push(cmd);
        Ok(())
    }
    
    /// Parse batch commands. Lebih efisien untuk kirim banyak perintah sekaligus.
    pub fn parse_batch(&mut self, json: &str) -> Result<(), serde_json::Error> {
        let cmds: Vec<<Command> = serde_json::from_str(json)?;
        self.queue.extend(cmds);
        Ok(())
    }
    
    /// Eksekusi semua queued command ke field & config
    pub fn execute(&mut self, field: &mut AgentField, config: &mut KernelConfig) {
        for cmd in self.queue.drain(..) {
            match cmd {
                Command::Spawn { x, y, vx, vy, health, color: _ } => {
                    let idx = field.spawn(x, y, health);
                    if idx < field.vel_x.len() {
                        field.vel_x[idx] = vx;
                        field.vel_y[idx] = vy;
                    }
                }
                Command::Kill { idx } => field.kill_swap(idx),
                Command::Config { dt, friction, max_speed, influence_radius } => {
                    *config = KernelConfig {
                        dt,
                        friction,
                        max_speed,
                        influence_radius,
                        ..*config
                    };
                }
                Command::Clear => {
                    field.len = 0;
                }
                Command::Batch(batch) => {
                    // Flatten: masukkan kembali ke queue untuk dieksekusi di iterasi berikutnya
                    self.queue.extend(batch);
                }
            }
        }
    }
    
    pub fn clear(&mut self) {
        self.queue.clear();
    }
}

