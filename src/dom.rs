use web_sys::{window, Document};

/// A utility module exposing safe wrappers around `web-sys` DOM manipulation.
/// This allows the Rust kernel (and subsequently Rhai scripts) to manipulate
/// the webpage directly without needing JavaScript glue code.
pub struct DomContext {
    document: Document,
}

impl DomContext {
    pub fn new() -> Option<Self> {
        let win = window()?;
        let document = win.document()?;
        Some(Self { document })
    }

    /// Appends raw HTML to a specific container ID.
    pub fn append_html(&self, target_id: &str, html: &str) -> Result<(), String> {
        if let Some(el) = self.document.get_element_by_id(target_id) {
            el.insert_adjacent_html("beforeend", html)
                .map_err(|_| "Failed to insert HTML".to_string())?;
            Ok(())
        } else {
            Err(format!("Element with id '{}' not found", target_id))
        }
    }

    /// Replaces the inner HTML of a specific container ID.
    pub fn set_inner_html(&self, target_id: &str, html: &str) -> Result<(), String> {
        if let Some(el) = self.document.get_element_by_id(target_id) {
            el.set_inner_html(html);
            Ok(())
        } else {
            Err(format!("Element with id '{}' not found", target_id))
        }
    }

    /// Updates text content of a specific container ID.
    pub fn set_text_content(&self, target_id: &str, text: &str) -> Result<(), String> {
         if let Some(el) = self.document.get_element_by_id(target_id) {
            el.set_text_content(Some(text));
            Ok(())
        } else {
            Err(format!("Element with id '{}' not found", target_id))
        }
    }
}
