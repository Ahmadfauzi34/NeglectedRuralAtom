use super::soa::AgentField;
use super::spatial_grid::SpatialGrid;
use super::environment_grid::EnvironmentGrid;

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
pub fn step_agents(field: &mut AgentField, config: &KernelConfig, grid: &mut SpatialGrid, env: &mut EnvironmentGrid) {
    let count = field.len;
    if count == 0 { return; }
    
    let dt = config.dt;
    let friction = config.friction;
    let max_speed_sq = config.max_speed * config.max_speed;
    let inf_r = config.influence_radius;
    let inf_r_sq = inf_r * inf_r;
    
    grid.set_cell_size(inf_r);
    grid.clear();

    for i in 0..count {
        if field.active[i] == 1 {
            grid.insert(field.pos_x[i], field.pos_y[i], i);
        }
    }

    // Pre-alloc accumulators di stack (bukan heap) — fixed size untuk branchless
    let mut acc_x = vec![0.0f32; count]; // ini alloc tapi di luar hot loop
    let mut acc_y = vec![0.0f32; count]; // untuk production, pakai pre-allocated scratch buffer
    
    // Scratch buffer for neighbor querying to avoid per-agent allocation
    let mut neighbors_buf = Vec::with_capacity(64);

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
        
        grid.query_neighbors(px, py, inf_r, &mut neighbors_buf);

        for &j in &neighbors_buf {
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
    
    // === PASS 2: State Machine Execution & Integration ===
    for i in 0..count {
        if field.active[i] == 0 { continue; }
        
        // Fetch explicit acceleration from boids physics
        let mut ax = acc_x[i];
        let mut ay = acc_y[i];

        // Apply behavior state overrides
        // 0 = Idle/Boids, 1 = Flee Center, 2 = Wander, 3 = Follow Environment Gradient (Pheromones)
        match field.behavior_state[i] {
            0 => { /* Normal Boids Physics */ },
            1 => {
                // Force flee from origin (0, 0)
                let px = field.pos_x[i];
                let py = field.pos_y[i];
                let dist = (px * px + py * py).sqrt() + 1e-6;
                ax += (px / dist) * 100.0;
                ay += (py / dist) * 100.0;
            },
            2 => {
                // Slight random wander (pseudo-random via modulo to keep WASM fast without RNG seeds)
                let pseudo_noise = (i as f32 * 0.1).sin();
                let pseudo_noise2 = (i as f32 * 0.1).cos();
                ax += pseudo_noise * 50.0;
                ay += pseudo_noise2 * 50.0;
            },
            3 => {
                // Pheromone/Gradient Navigation
                let px = field.pos_x[i];
                let py = field.pos_y[i];
                let s = env.cell_size;

                // Sample 4 cardinal directions to find the steepest gradient
                let v_up = env.read_value(px, py - s);
                let v_down = env.read_value(px, py + s);
                let v_left = env.read_value(px - s, py);
                let v_right = env.read_value(px + s, py);

                let grad_x = v_right - v_left;
                let grad_y = v_down - v_up;

                // Move towards highest pheromone concentration
                ax += grad_x * 5.0; // Pheromone attraction strength
                ay += grad_y * 5.0;
            },
            _ => {}
        }

        // Update velocity
        field.vel_x[i] += ax;
        field.vel_y[i] += ay;
        
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
