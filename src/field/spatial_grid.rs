use std::collections::HashMap;

/// A simple Spatial Hash Grid to accelerate 2D neighbor searches.
/// Reduces O(N^2) complexity to roughly O(N) by dividing space into cells.
pub struct SpatialGrid {
    cell_size: f32,
    cells: HashMap<u64, Vec<usize>>,
    // We can avoid HashMap allocations per frame by reusing cell vectors.
    // In a fully optimized engine, we'd use a flat array + sorting or linked lists per cell.
    // For simplicity and immediate O(N) speedup over O(N^2), HashMap is used here.
}

impl SpatialGrid {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            cells: HashMap::with_capacity(1024),
        }
    }

    pub fn set_cell_size(&mut self, size: f32) {
        self.cell_size = size.max(1.0);
    }

    pub fn clear(&mut self) {
        // Clear inner vectors but keep capacity to avoid reallocation
        for cell in self.cells.values_mut() {
            cell.clear();
        }
    }

    #[inline(always)]
    fn get_cell_key(&self, x: f32, y: f32) -> u64 {
        // Map 2D float coordinates to 1D integer grid index
        // Using a large offset to handle negative coordinates smoothly
        let ix = (x / self.cell_size).floor() as i32;
        let iy = (y / self.cell_size).floor() as i32;

        let ux = (ix as u32) as u64;
        let uy = (iy as u32) as u64;

        (ux << 32) | uy
    }

    pub fn insert(&mut self, x: f32, y: f32, id: usize) {
        let key = self.get_cell_key(x, y);
        self.cells
            .entry(key)
            .or_insert_with(|| Vec::with_capacity(16))
            .push(id);
    }

    /// Queries the grid for neighbors around a coordinate within a radius.
    /// Fills the provided `out` buffer with agent IDs to avoid allocation.
    pub fn query_neighbors(&self, x: f32, y: f32, radius: f32, out: &mut Vec<usize>) {
        out.clear();

        // Find bounding box cells
        let min_x = x - radius;
        let max_x = x + radius;
        let min_y = y - radius;
        let max_y = y + radius;

        let start_ix = (min_x / self.cell_size).floor() as i32;
        let end_ix = (max_x / self.cell_size).floor() as i32;
        let start_iy = (min_y / self.cell_size).floor() as i32;
        let end_iy = (max_y / self.cell_size).floor() as i32;

        for ix in start_ix..=end_ix {
            for iy in start_iy..=end_iy {
                let ux = (ix as u32) as u64;
                let uy = (iy as u32) as u64;
                let key = (ux << 32) | uy;

                if let Some(cell) = self.cells.get(&key) {
                    out.extend(cell);
                }
            }
        }
    }
}
