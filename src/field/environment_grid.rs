/// Represents a 2D Voxel/Sensor grid over the simulation field.
/// Useful for simulating pheromones, heatmaps, obstacles, or LLM-painted goals.
pub struct EnvironmentGrid {
    pub(crate) cells: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub cell_size: f32,
}

impl EnvironmentGrid {
    pub fn new(width: usize, height: usize, cell_size: f32) -> Self {
        Self {
            cells: vec![0.0; width * height],
            width,
            height,
            cell_size,
        }
    }

    /// Converts continuous 2D coordinates into discrete 1D cell index.
    #[inline(always)]
    fn get_index(&self, x: f32, y: f32) -> Option<usize> {
        let ix = (x / self.cell_size).floor() as i32;
        let iy = (y / self.cell_size).floor() as i32;

        if ix >= 0 && ix < (self.width as i32) && iy >= 0 && iy < (self.height as i32) {
            Some((iy as usize) * self.width + (ix as usize))
        } else {
            None
        }
    }

    /// Reads the scalar value at a given position.
    pub fn read_value(&self, x: f32, y: f32) -> f32 {
        if let Some(idx) = self.get_index(x, y) {
            self.cells[idx]
        } else {
            0.0 // Outside grid boundaries is considered zero presence
        }
    }

    /// Directly overwrites the value at a given position.
    pub fn set_value(&mut self, x: f32, y: f32, value: f32) {
        if let Some(idx) = self.get_index(x, y) {
            self.cells[idx] = value;
        }
    }

    /// Adds a value to a given position (useful for continuous pheromone dropping).
    pub fn add_value(&mut self, x: f32, y: f32, amount: f32) {
        if let Some(idx) = self.get_index(x, y) {
            self.cells[idx] += amount;
        }
    }

    /// Multiplies every cell by a decay factor (e.g., 0.99) to simulate
    /// evaporating smells or fading heatmaps over time.
    /// Clamps very small values to 0.0 to prevent subnormal float slowdowns.
    pub fn decay(&mut self, factor: f32) {
        for val in self.cells.iter_mut() {
            if *val > 0.001 {
                *val *= factor;
            } else {
                *val = 0.0;
            }
        }
    }

    /// Clears the entire grid back to zero.
    pub fn clear(&mut self) {
        self.cells.fill(0.0);
    }
}
