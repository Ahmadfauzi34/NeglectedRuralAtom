use super::soa::AgentField;

/// Config untuk kernel — internal Rust, tidak di-expose ke JS
/// (parameter di-pass via primitive di KernelBridge::set_config)
#[derive(Clone, Copy)]
pub struct KernelConfig {
    pub dt: f32,
    pub friction: f32,
    pub max_speed: f32,
    pub influence_radius: f32,
    pub separation_weight: f32,
    pub alignment_weight: f32,
    pub cohesion_weight: f32,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            dt: 0.016,
            friction: 0.95,
            max_speed: 200.0,
            influence_radius: 80.0,
            separation_weight: 1.5,
            alignment_weight: 1.0,
            cohesion_weight: 1.0,
        }
    }
}

/// Step simulation — hot path, zero allocation, branchless where possible
pub fn step_agents(field: &mut AgentField, config: &KernelConfig) {
    let count = field.len;
    if count == 0 { return; }
    
    let dt = config.dt;
    let friction = config.friction;
    let max_speed_sq = config.max_speed * config.max_speed;
    let inf_r = config.influence_radius;
    let inf_r_sq = inf_r * inf_r;
    
    // Pre-alloc accumulators di stack (bukan heap) — fixed size untuk branchless
    let mut acc_x = vec![0.0f32; count]; // ini alloc tapi di luar hot loop
    let mut acc_y = vec![0.0f32; count]; // untuk production, pakai pre-allocated scratch buffer
    
    // === PASS 1: Field influence computation (continuous query) ===
    for i in 0..count {
        if field.active[i] == 0 { continue; }
        
        let px = field.pos_x[i];
        let py = field.pos_y[i];
        
        let mut sep_x = 0.0f32;
        let mut sep_y = 0.0f32;
        let mut ali_x = 0.0f32;
        let mut ali_y = 0.0f32;
        let mut coh_x = 0.0f32;
        let mut coh_y = 0.0f32;
        let mut neighbors = 0u32;
        
        for j in 0..count {
            if i == j || field.active[j] == 0 { continue; }
            
            let dx = field.pos_x[j] - px;
            let dy = field.pos_y[j] - py;
            let dist_sq = dx * dx + dy * dy;
            
            // Branchless: gunakan mask alih-alih if
            let in_range = (dist_sq <= inf_r_sq) as i32 as f32; // 1.0 atau 0.0
            let inv_dist = in_range / (dist_sq + 1e-6); // epsilon untuk avoid div by zero
            
            // Separation: steer away
            sep_x += -dx * inv_dist * in_range;
            sep_y += -dy * inv_dist * in_range;
            
            // Alignment: match velocity
            ali_x += field.vel_x[j] * in_range;
            ali_y += field.vel_y[j] * in_range;
            
            // Cohesion: steer to center
            coh_x += dx * in_range;
            coh_y += dy * in_range;
            
            neighbors += in_range as u32;
        }
        
        // Apply weights — branchless dengan neighbor count
        let n = neighbors.max(1) as f32;
        let n_inv = 1.0 / n;
        
        acc_x[i] = (sep_x * config.separation_weight + 
                    ali_x * config.alignment_weight * n_inv + 
                    coh_x * config.cohesion_weight * n_inv) * dt;
                    
        acc_y[i] = (sep_y * config.separation_weight + 
                    ali_y * config.alignment_weight * n_inv + 
                    coh_y * config.cohesion_weight * n_inv) * dt;
    }
    
    // === PASS 2: Integration (SOA iteration — cache optimal) ===
    for i in 0..count {
        if field.active[i] == 0 { continue; }
        
        // Update velocity
        field.vel_x[i] += acc_x[i];
        field.vel_y[i] += acc_y[i];
        
        // Apply friction
        field.vel_x[i] *= friction;
        field.vel_y[i] *= friction;
        
        // Clamp speed — branchless menggunakan min/max
        let vx = field.vel_x[i];
        let vy = field.vel_y[i];
        let speed_sq = vx * vx + vy * vy;
        
        // Kalau speed_sq > max_speed_sq, scale down — branchless
        let scale = (speed_sq > max_speed_sq) as i32 as f32;
        let speed = (speed_sq + 1e-6).sqrt();
        let factor = 1.0 - scale + scale * (config.max_speed / speed);
        
        field.vel_x[i] = vx * factor;
        field.vel_y[i] = vy * factor;
        
        // Update position
        field.pos_x[i] += field.vel_x[i] * dt;
        field.pos_y[i] += field.vel_y[i] * dt;
    }
}
