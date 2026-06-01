use std::collections::HashMap;
use regex::Regex;

#[derive(Default, Debug, Clone)]
pub struct VirtualFileSystem {
    pub files: HashMap<String, String>,
}

impl VirtualFileSystem {
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
        }
    }

    pub fn write_file(&mut self, path: &str, content: &str) {
        // Anti-memory-leak: Cap number of files and file size
        if self.files.len() >= 1024 && !self.files.contains_key(path) {
            return; // File system full
        }

        let mut safe_path = path.to_string();
        if safe_path.len() > 256 {
            let mut end = 256;
            while end > 0 && !safe_path.is_char_boundary(end) {
                end -= 1;
            }
            safe_path.truncate(end);
        }

        let mut safe_content = content.to_string();
        if safe_content.len() > 1024 * 512 { // Max 512 KB per file
            let mut end = 1024 * 512;
            while end > 0 && !safe_content.is_char_boundary(end) {
                end -= 1;
            }
            safe_content.truncate(end);
        }

        self.files.insert(safe_path, safe_content);
    }

    pub fn read_file(&self, path: &str) -> Option<String> {
        self.files.get(path).cloned()
    }

    pub fn edit_file(&mut self, path: &str, append_content: &str) {
        // Anti-memory-leak: Cap number of files
        if self.files.len() >= 1024 && !self.files.contains_key(path) {
            return;
        }

        let mut safe_path = path.to_string();
        if safe_path.len() > 256 {
            let mut end = 256;
            while end > 0 && !safe_path.is_char_boundary(end) {
                end -= 1;
            }
            safe_path.truncate(end);
        }

        let current = self.files.entry(safe_path).or_insert_with(String::new);

        // Prevent unbounded file growth via continuous appending (UTF-8 safe)
        if current.len() + append_content.len() > 1024 * 512 {
            let mut space_left = (1024_usize * 512).saturating_sub(current.len());
            if space_left > 0 {
                if space_left > append_content.len() {
                    space_left = append_content.len();
                }
                while space_left > 0 && !append_content.is_char_boundary(space_left) {
                    space_left -= 1;
                }
                current.push_str(&append_content[..space_left]);
            }
        } else {
            current.push_str(append_content);
        }
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
