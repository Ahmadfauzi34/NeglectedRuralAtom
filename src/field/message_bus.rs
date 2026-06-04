use std::collections::HashMap;

/// Constant to define a broadcast message meant for all agents
pub const BROADCAST_ID: u32 = u32::MAX;

/// Structure of Arrays (SOA) Message Bus for high-performance inter-agent communication.
/// It uses a centralized text arena to prevent heap fragmentation during massive message bursts.
pub struct MessageBus {
    pub(crate) sender_ids: Vec<u32>,
    pub(crate) receiver_ids: Vec<u32>,
    pub(crate) message_types: Vec<u8>,

    // Contiguous memory arena for string payloads
    pub(crate) text_arena: String,
    pub(crate) payload_slices: Vec<(u32, u32)>,

    // O(1) Indexing for rapid queries
    pub(crate) receiver_index: HashMap<u32, Vec<usize>>,

    pub(crate) len: usize,
    pub(crate) max_arena_bytes: usize,
}

impl MessageBus {
    pub fn new(initial_capacity: usize) -> Self {
        Self {
            sender_ids: Vec::with_capacity(initial_capacity),
            receiver_ids: Vec::with_capacity(initial_capacity),
            message_types: Vec::with_capacity(initial_capacity),
            text_arena: String::with_capacity(initial_capacity * 64),
            payload_slices: Vec::with_capacity(initial_capacity),
            receiver_index: HashMap::with_capacity(initial_capacity),
            len: 0,
            // Dynamic capacity based on initial setup.
            // Ensures messages can scale linearly with agents, while bounding runaway loops.
            max_arena_bytes: initial_capacity * 2048,
        }
    }

    /// Sends a message from one agent to another (or broadcast if receiver_id == BROADCAST_ID).
    pub fn send_message(&mut self, sender_id: u32, receiver_id: u32, msg_type: u8, payload: &str) {
        // Anti-memory-leak: Prevent unbounded vector structural growth.
        // Max messages dynamically linked to arena bytes (assuming avg ~32 bytes per message)
        let structural_max_messages = self.max_arena_bytes / 32;
        if self.sender_ids.len() >= structural_max_messages {
            return;
        }

        let mut safe_payload = payload;

        // Anti-memory-leak: Dynamically cap the global text arena.
        // Drops message entirely if 0 bytes available, else gracefully truncates.
        if self.text_arena.len() + safe_payload.len() > self.max_arena_bytes {
            let available = self.max_arena_bytes.saturating_sub(self.text_arena.len());
            if available == 0 && payload.len() > 0 {
                return; // Dropped to prevent memory ballooning
            }
            let mut end = available;
            if end > safe_payload.len() {
                end = safe_payload.len();
            }
            while end > 0 && !safe_payload.is_char_boundary(end) {
                end -= 1;
            }
            safe_payload = &safe_payload[..end];
        }

        let p_start = self.text_arena.len() as u32;
        self.text_arena.push_str(safe_payload);
        let p_end = self.text_arena.len() as u32;

        let current_idx = self.len;

        self.sender_ids.push(sender_id);
        self.receiver_ids.push(receiver_id);
        self.message_types.push(msg_type);
        self.payload_slices.push((p_start, p_end));

        // Populate fast lookup index
        self.receiver_index
            .entry(receiver_id)
            .or_insert_with(Vec::new)
            .push(current_idx);

        self.len += 1;
    }

    /// Retrieves all messages meant for a specific receiver.
    /// Includes point-to-point and broadcast messages via O(1) lookup.
    /// Fills the provided output buffer with indices of the matching messages.
    pub fn query_messages(&self, receiver_id: u32, out_indices: &mut Vec<usize>) {
        out_indices.clear();

        // 1. Fetch exact matches
        if let Some(indices) = self.receiver_index.get(&receiver_id) {
            out_indices.extend_from_slice(indices);
        }

        // 2. Fetch broadcasts
        if receiver_id != BROADCAST_ID {
            if let Some(broadcasts) = self.receiver_index.get(&BROADCAST_ID) {
                out_indices.extend_from_slice(broadcasts);
            }
        }
    }

    /// Retrieves the text payload for a specific message index.
    pub fn get_payload(&self, idx: usize) -> &str {
        if idx < self.len {
            let (start, end) = self.payload_slices[idx];
            &self.text_arena[start as usize..end as usize]
        } else {
            ""
        }
    }

    /// Flushes all messages. Typically called at the start or end of every simulation tick
    /// to ensure messages are transient (single-frame lifespan).
    pub fn clear(&mut self) {
        self.sender_ids.clear();
        self.receiver_ids.clear();
        self.message_types.clear();
        self.payload_slices.clear();
        self.text_arena.clear();
        self.receiver_index.clear();
        self.len = 0;
    }
}
