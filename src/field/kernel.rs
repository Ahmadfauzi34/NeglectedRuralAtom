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
    pub cursor_x: f32,
    pub cursor_y: f32,
    pub cursor_weight: f32,
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
            cursor_x: 0.0,
            cursor_y: 0.0,
            cursor_weight: 0.0,
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

    // Reset pre-allocated accumulators without dropping capacity
    field.acc_x[..count].fill(0.0);
    field.acc_y[..count].fill(0.0);

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
        
        field.acc_x[i] = (sep_x * config.separation_weight +
                          ali_x * config.alignment_weight * n_inv +
                          coh_x * config.cohesion_weight * n_inv) * dt;
                    
        field.acc_y[i] = (sep_y * config.separation_weight +
                          ali_y * config.alignment_weight * n_inv +
                          coh_y * config.cohesion_weight * n_inv) * dt;
    }
    
    // === PASS 2: State Machine Execution & Integration ===
    for i in 0..count {
        if field.active[i] == 0 { continue; }
        
        // Fetch explicit acceleration from boids physics
        let mut ax = field.acc_x[i];
        let mut ay = field.acc_y[i];

        let px = field.pos_x[i];
        let py = field.pos_y[i];

        // --- 1. Cursor Gravity / Attraction ---
        if config.cursor_weight.abs() > 0.001 {
            let dx = config.cursor_x - px;
            let dy = config.cursor_y - py;
            let dist_sq = dx * dx + dy * dy;
            let dist = dist_sq.sqrt() + 1e-6;
            // Force is inversely proportional to distance, capped max
            let force = (1000.0 / dist).min(50.0) * config.cursor_weight;
            ax += (dx / dist) * force;
            ay += (dy / dist) * force;
        }

        // --- 2. Obstacle Avoidance (Negative Pheromone Repulsion) ---
        // Reads the environment grid at agent's position. If negative (obstacle), push away rapidly.
        let local_env = env.read_value(px, py);
        if local_env < 0.0 {
            // Find quickest way out by checking gradients
            let s = env.cell_size;
            let v_up = env.read_value(px, py - s);
            let v_down = env.read_value(px, py + s);
            let v_left = env.read_value(px - s, py);
            let v_right = env.read_value(px + s, py);

            // Move opposite to the steepest negative descent
            let grad_x = v_right - v_left;
            let grad_y = v_down - v_up;

            // Repulsion scale proportional to how deep into the obstacle they are
            let repulse_force = local_env.abs() * 20.0;
            ax += -grad_x * repulse_force;
            ay += -grad_y * repulse_force;
        }

        // Apply behavior state overrides
        // 0 = Idle/Boids, 1 = Flee Center, 2 = Wander, 3 = Follow Environment Gradient (Pheromones), 4 = Predator, 5 = Prey
        match field.behavior_state[i] {
            0 => { /* Normal Boids Physics */ },
            1 => {
                // Force flee from origin (0, 0)
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
            4 => {
                // Predator: Find nearest prey (State 5) and chase it
                grid.query_neighbors(px, py, inf_r * 2.0, &mut neighbors_buf);
                let mut closest_dist_sq = f32::MAX;
                let mut target_dx = 0.0;
                let mut target_dy = 0.0;

                for &j in &neighbors_buf {
                    if i == j || field.active[j] == 0 { continue; }
                    if field.behavior_state[j] == 5 { // Is Prey
                        let dx = field.pos_x[j] - px;
                        let dy = field.pos_y[j] - py;
                        let dist_sq = dx * dx + dy * dy;
                        if dist_sq < closest_dist_sq {
                            closest_dist_sq = dist_sq;
                            target_dx = dx;
                            target_dy = dy;
                        }
                    }
                }

                if closest_dist_sq < f32::MAX {
                    let dist = closest_dist_sq.sqrt() + 1e-6;
                    ax += (target_dx / dist) * 150.0; // Strong chase force
                    ay += (target_dy / dist) * 150.0;
                }
            },
            5 => {
                // Prey: Find nearest predator (State 4) and flee
                grid.query_neighbors(px, py, inf_r * 2.0, &mut neighbors_buf);
                for &j in &neighbors_buf {
                    if i == j || field.active[j] == 0 { continue; }
                    if field.behavior_state[j] == 4 { // Is Predator
                        let dx = field.pos_x[j] - px;
                        let dy = field.pos_y[j] - py;
                        let dist_sq = dx * dx + dy * dy;
                        let dist = dist_sq.sqrt() + 1e-6;
                        // Exponentially strong flee force the closer the predator is
                        let flee_force = (inf_r * 2.0) / dist * 50.0;
                        ax -= (dx / dist) * flee_force;
                        ay -= (dy / dist) * flee_force;
                    }
                }
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
