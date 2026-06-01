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
        self.files.insert(path.to_string(), content.to_string());
    }

    pub fn read_file(&self, path: &str) -> Option<String> {
        self.files.get(path).cloned()
    }

    pub fn edit_file(&mut self, path: &str, append_content: &str) {
        let current = self.files.entry(path.to_string()).or_insert_with(String::new);
        current.push_str(append_content);
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
