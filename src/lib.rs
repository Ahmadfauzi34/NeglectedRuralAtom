#![cfg_attr(not(test), warn(clippy::all, clippy::pedantic, clippy::cargo))]
// ==========================================
// ⛔ STRICT DENY (Keamanan & Anti-Mangkir)
// ==========================================
#![cfg_attr(not(test), deny(
    clippy::correctness,
    clippy::suspicious,
    clippy::unwrap_used,   // Wajib handle error (jangan pakai panics)
    clippy::expect_used,   // Sama seperti unwrap
    clippy::todo,          // Cegah AI/Developer meninggalkan placeholder
    clippy::unimplemented, // Cegah fungsi kosong masuk ke production
))]
// ==========================================
// 🚧 TEMPORARY ALLOW (Tersisa Prioritas Merah & Struktural)
// ==========================================
#![allow(
    clippy::suboptimal_flops,      // 🔴 Paling tinggi: tensor math (Belum dioptimasi)

    // ⚠️ STRUKTURAL: Dipertahankan agar AI tidak merusak arsitektur hot-path SoA
    clippy::too_many_lines,
    clippy::too_many_arguments,
)]
// ==========================================
// 🛡️ PERMANENT ALLOW (Domain Tensor)
// ==========================================
#![allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::unreadable_literal, // FHRR random generator constant
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
)]
mod bridge;
pub mod business;
mod command;
pub mod dom;
mod field;
pub mod graph;
pub mod prompt;
mod render;
pub mod scripting;
pub mod svg_generator;
pub mod telemetry;

use wasm_bindgen::prelude::*;

#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

pub use bridge::KernelBridge;
pub use command::CommandBus;
pub use field::{step_agents, AgentField, KernelConfig};
pub use render::{
    CanvasEncoder, DrawCmd, DrawHeader, TAG_CIRCLE, TAG_LINE, TAG_POLY, TAG_RECT, TAG_TEXT,
};
pub mod vfs;
use crate::vfs::VirtualFileSystem;
