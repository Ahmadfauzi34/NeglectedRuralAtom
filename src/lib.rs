mod field;
mod bridge;
mod command;
mod render;
pub mod scripting;
pub mod dom;
pub mod prompt;
pub mod business;

use wasm_bindgen::prelude::*;

#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

pub use field::{AgentField, KernelConfig, step_agents};
pub use bridge::{KernelBridge, MemoryView};
pub use command::CommandBus;
pub use render::{CanvasEncoder, DrawHeader, DrawCmd, TAG_CIRCLE, TAG_LINE, TAG_POLY, TAG_RECT, TAG_TEXT};
