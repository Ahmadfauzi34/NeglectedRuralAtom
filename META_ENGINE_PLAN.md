# META SCRIPT ENGINE v2.0 - IMPLEMENTATION ROADMAP

## OVERVIEW
Transforming the existing `ScriptEngine` into a multi-layered, highly optimized execution engine for Recursive LLM Agents.

## PHASES

### PHASE 1: Preparation & Layer 3 (Global AST Pool)
- [ ] Add `seahash` to `Cargo.toml`.
- [ ] Create `src/scripting/meta_optimizer.rs` and update `src/scripting/mod.rs`.
- [ ] Implement `MetaEngineConfig` and `MetaTelemetry`.
- [ ] Implement `GlobalAstPool` (L3) with LRU eviction tracking based on frame ticks.
- [ ] Refactor existing `ScriptEngine` to wrap `MetaScriptEngine` and route `eval` calls to use the L3 cache.

### PHASE 2: Layer 1 (Agent Exact Cache & Context Fingerprint)
- [ ] Implement `ContextFingerprint` (hashing behavior, pos, health).
- [ ] Implement `AgentCache` mapping `ExactCacheKey` to `CachedExecution`.
- [ ] Add TTL tracking based on agent behavior states.
- [ ] Implement `eval_for_agent` in `KernelBridge`.

### PHASE 3: Layer 2 (Fuzzy Pattern Matching via SimHash)
- [ ] Implement `SimHash` feature hashing (64-bit fingerprinting of script text chunks).
- [ ] Implement `PatternIndex` with Hamming distance queries.
- [ ] Integrate L2 checks between L1 and L3.
- [ ] Add auto-optimization (telemetry-driven tuning).

### PHASE 4: Swarm APIs & Self-Modification
- [ ] Implement `eval_broadcast` (Rust) and `eval_swarm_script` (JS).
- [ ] Implement cache invalidation hooks (`forget_script`, `forget_agent`, `forget_all`).
- [ ] Finalize telemetry JSON exports and config updates from JS.
- [ ] Extensive testing & stabilization.
