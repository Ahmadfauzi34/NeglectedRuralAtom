mod field;
mod bridge;

use wasm_bindgen::prelude::*;

// Allocator kecil untuk WASM
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

// Init panic hook untuk debug (bisa dihapus di production untuk ukuran lebih kecil)
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

// Re-export untuk JS
pub use field::{AgentField, KernelConfig};
pub use bridge::{KernelBridge, MemoryView};
