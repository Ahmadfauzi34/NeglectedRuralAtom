use std::collections::HashMap;
use regex::Regex;

// Maximum total capacity for the Virtual File System (e.g. 64 MB)
// This allows dynamic files (some very large, some very small) without arbitrary file count limits.
const MAX_VFS_CAPACITY_BYTES: usize = 64 * 1024 * 1024;

#[derive(Default, Debug, Clone)]
pub struct VirtualFileSystem {
    pub files: HashMap<String, String>,
    pub total_bytes: usize,
}

impl VirtualFileSystem {
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
            total_bytes: 0,
        }
    }

    pub fn write_file(&mut self, path: &str, content: &str) {
        let mut safe_path = path.to_string();
        if safe_path.len() > 256 {
            let mut end = 256;
            while end > 0 && !safe_path.is_char_boundary(end) {
                end -= 1;
            }
            safe_path.truncate(end);
        }

        // Anti-memory-leak: Prevent unbound HashMap structural growth.
        // A dynamic structural limit based on total bytes (assume avg 4KB per file + 128 keys)
        let max_structural_files = MAX_VFS_CAPACITY_BYTES / 4096;
        if self.files.len() >= max_structural_files && !self.files.contains_key(&safe_path) {
            return;
        }

        // Calculate size difference if file already exists
        let old_size = self.files.get(&safe_path).map_or(0, |s| s.len());
        let new_size = content.len();

        let projected_total = (self.total_bytes - old_size) + new_size;

        // Anti-memory-leak: Cap global VFS size dynamically
        if projected_total > MAX_VFS_CAPACITY_BYTES {
            // If it exceeds global quota, we try to fit as much as we safely can
            let available_space = MAX_VFS_CAPACITY_BYTES.saturating_sub(self.total_bytes - old_size);
            if available_space == 0 {
                return; // No space left
            }

            let mut end = available_space;
            if end > content.len() {
                end = content.len();
            }
            while end > 0 && !content.is_char_boundary(end) {
                end -= 1;
            }

            let safe_content = content[..end].to_string();
            self.total_bytes = (self.total_bytes - old_size) + safe_content.len();
            self.files.insert(safe_path, safe_content);
        } else {
            self.total_bytes = projected_total;
            self.files.insert(safe_path, content.to_string());
        }
    }

    pub fn read_file(&self, path: &str) -> Option<String> {
        self.files.get(path).cloned()
    }

    pub fn edit_file(&mut self, path: &str, append_content: &str) {
        let mut safe_path = path.to_string();
        if safe_path.len() > 256 {
            let mut end = 256;
            while end > 0 && !safe_path.is_char_boundary(end) {
                end -= 1;
            }
            safe_path.truncate(end);
        }

        // Anti-memory-leak: Prevent unbound HashMap structural growth.
        let max_structural_files = MAX_VFS_CAPACITY_BYTES / 4096;
        if self.files.len() >= max_structural_files && !self.files.contains_key(&safe_path) {
            return;
        }

        let projected_total = self.total_bytes + append_content.len();

        // Anti-memory-leak: Cap global VFS size dynamically on append
        let content_to_append = if projected_total > MAX_VFS_CAPACITY_BYTES {
            let available_space = MAX_VFS_CAPACITY_BYTES.saturating_sub(self.total_bytes);
            if available_space == 0 {
                return;
            }

            let mut end = available_space;
            if end > append_content.len() {
                end = append_content.len();
            }
            while end > 0 && !append_content.is_char_boundary(end) {
                end -= 1;
            }
            &append_content[..end]
        } else {
            append_content
        };

        if content_to_append.is_empty() {
            return;
        }

        let current = self.files.entry(safe_path).or_insert_with(String::new);
        current.push_str(content_to_append);
        self.total_bytes += content_to_append.len();
    }

    pub fn list_directory(&self, prefix: &str) -> Vec<String> {
        self.files
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect()
    }

    pub fn search_files(&self, pattern: &str) -> Vec<String> {
        if let Ok(regex) = Regex::new(pattern) {
            self.files
                .keys()
                .filter(|k| regex.is_match(k))
                .cloned()
                .collect()
        } else {
            vec![]
        }
    }
}
