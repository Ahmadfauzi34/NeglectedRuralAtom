//! Recursive Agent Architecture
//!
//! Layer hierarchy:
//! - WorkerAgent: Execute + local learning (Layer 1)
//! - OrchestratorAgent: Decompose + route + merge (Layer 2)
//! - MetaAgent: Self-modify + evolve + spawn (Layer 3)

pub mod worker;
pub mod orchestrator;
pub mod meta;

pub use worker::WorkerAgent;
pub use orchestrator::OrchestratorAgent;
pub use meta::MetaAgent;
