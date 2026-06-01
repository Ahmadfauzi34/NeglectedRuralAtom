# Agentic Kernel

A highly optimized, zero-allocation WebAssembly (WASM) kernel designed to empower Client-Side Artificial Intelligence and LLM-driven Swarm logic.

## Features

- **Asymmetric TRM Core:** Implements Sparse Tensor mathematical operations (Spectral Processing, Cross-Attention Bridges, and Orthogonal Projection) natively in Rust.
- **Agentic Autonomy:** Empowers `DataWorker` swarms to independently process data via Multi-Layer Perceptrons and Contextual Meta-Learning (`q_learning_update`, `context_evolution`), radically reducing heavy JSON API round-trips to the LLM.
- **RAG & Vector Memory:** A fixed-dimensional micro vector database computing cosine-similarities natively for real-time memory injection.
- **Graph Pipeline Executor:** Evaluates LLM-injected logic via Rhai scripts with AST caching and parallel DAG branching.
- **DOM & SVG Visualizations:** Binds Rhai safely to the DOM, HTML Canvas 2D (`HtmlCanvasElement`), and generates intricate `<foreignObject>` SVG charts using `plotters`.
- **Zero-Allocation Architecture:** Entire codebase follows Structure of Arrays (SoA), memory arenas, and defensive boundaries.

## Build Instructions

Requires `wasm-pack` and the `wasm32-unknown-unknown` target.

```bash
cargo build --lib --release --target wasm32-unknown-unknown
wasm-pack build --target web
```
