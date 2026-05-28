use super::CanvasEncoder;
use crate::field::soa::AgentField;

/// System: encode agent field ke draw commands.
/// Bisa diganti/ditambah system lain (particle renderer, graph renderer) tanpa ubah TS.
pub fn encode_agents(encoder: &mut CanvasEncoder, field: &AgentField, default_color: u32) {
    encoder.clear();
    
    for i in 0..field.len {
        if field.active[i] == 0 { continue; }
        
        let x = field.pos_x[i];
        let y = field.pos_y[i];
        let vx = field.vel_x[i];
        let vy = field.vel_y[i];
        
        // Agent body
        encoder.circle(x, y, 3.5, default_color);
        
        // Velocity vector (debug viz)
        let speed_sq = vx * vx + vy * vy;
        if speed_sq > 4.0 {
            let speed = speed_sq.sqrt();
            let nx = vx / speed;
            let ny = vy / speed;
            encoder.line(x, y, x + nx * 10.0, y + ny * 10.0, 0x60FFFFFF);
        }
    }
}

