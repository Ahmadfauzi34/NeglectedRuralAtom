use web_sys::{window, Document, HtmlElement, HtmlInputElement, HtmlCanvasElement, CanvasRenderingContext2d};
use wasm_bindgen::JsCast;

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

    /// Retrieves the inner HTML of a specific container ID.
    /// Useful for agents to compare the original content before making edits (Diffing).
    pub fn get_html(&self, target_id: &str) -> Result<String, String> {
        if let Some(el) = self.document.get_element_by_id(target_id) {
            Ok(el.inner_html())
        } else {
            Err(format!("Element with id '{}' not found", target_id))
        }
    }

    /// Appends raw HTML to a specific container ID at the end of its children.
    pub fn append_html(&self, target_id: &str, html: &str) -> Result<(), String> {
        self.insert_html_at(target_id, "beforeend", html)
    }

    /// Inserts raw HTML relative to a specific container ID.
    /// `position` can be: "beforebegin", "afterbegin", "beforeend", or "afterend".
    pub fn insert_html_at(&self, target_id: &str, position: &str, html: &str) -> Result<(), String> {
        if let Some(el) = self.document.get_element_by_id(target_id) {
            el.insert_adjacent_html(position, html)
                .map_err(|_| format!("Failed to insert HTML at position '{}'", position))?;
            Ok(())
        } else {
            Err(format!("Element with id '{}' not found", target_id))
        }
    }

    /// Finds a specific string within an element's inner HTML and replaces it with a new string.
    /// Returns true if a replacement occurred, false otherwise.
    pub fn diff_and_replace_html(&self, target_id: &str, old_str: &str, new_str: &str) -> Result<bool, String> {
        if let Some(el) = self.document.get_element_by_id(target_id) {
            let current_html = el.inner_html();
            if current_html.contains(old_str) {
                let replaced = current_html.replace(old_str, new_str);
                el.set_inner_html(&replaced);
                Ok(true)
            } else {
                Ok(false)
            }
        } else {
            Err(format!("Element with id '{}' not found", target_id))
        }
    }

    /// Replaces the inner HTML of a specific container ID entirely.
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

    /// Sets a CSS style property on a specific element.
    pub fn set_style(&self, target_id: &str, property: &str, value: &str) -> Result<(), String> {
        if let Some(el) = self.document.get_element_by_id(target_id) {
            if let Ok(html_el) = el.dyn_into::<HtmlElement>() {
                html_el.style().set_property(property, value)
                    .map_err(|_| format!("Failed to set style property '{}'", property))?;
                Ok(())
            } else {
                Err(format!("Element '{}' is not an HtmlElement", target_id))
            }
        } else {
            Err(format!("Element with id '{}' not found", target_id))
        }
    }

    /// Retrieves the current string value of an HTML input element.
    pub fn get_value(&self, target_id: &str) -> Result<String, String> {
        if let Some(el) = self.document.get_element_by_id(target_id) {
            if let Ok(input_el) = el.dyn_into::<HtmlInputElement>() {
                Ok(input_el.value())
            } else {
                Err(format!("Element '{}' is not an HtmlInputElement", target_id))
            }
        } else {
            Err(format!("Element with id '{}' not found", target_id))
        }
    }

    /// Directly calls a Canvas 2D Context API to fill a rectangle from WASM.
    pub fn canvas_fill_rect(&self, target_id: &str, x: f64, y: f64, w: f64, h: f64, fill_style: &str) -> Result<(), String> {
        if let Some(el) = self.document.get_element_by_id(target_id) {
            if let Ok(canvas) = el.dyn_into::<HtmlCanvasElement>() {
                if let Ok(Some(ctx_obj)) = canvas.get_context("2d") {
                    if let Ok(ctx) = ctx_obj.dyn_into::<CanvasRenderingContext2d>() {
                        #[allow(deprecated)]
                        ctx.set_fill_style(&fill_style.into());
                        ctx.fill_rect(x, y, w, h);
                        return Ok(());
                    }
                }
            }
            Err(format!("Element '{}' is not a valid Canvas 2D", target_id))
        } else {
            Err(format!("Element with id '{}' not found", target_id))
        }
    }

    /// Clears a Canvas 2D context from WASM.
    pub fn canvas_clear_rect(&self, target_id: &str, x: f64, y: f64, w: f64, h: f64) -> Result<(), String> {
        if let Some(el) = self.document.get_element_by_id(target_id) {
            if let Ok(canvas) = el.dyn_into::<HtmlCanvasElement>() {
                if let Ok(Some(ctx_obj)) = canvas.get_context("2d") {
                    if let Ok(ctx) = ctx_obj.dyn_into::<CanvasRenderingContext2d>() {
                        ctx.clear_rect(x, y, w, h);
                        return Ok(());
                    }
                }
            }
            Err(format!("Element '{}' is not a valid Canvas 2D", target_id))
        } else {
            Err(format!("Element with id '{}' not found", target_id))
        }
    }

    /// Fast Code Writer: Draws text directly onto a Canvas 2D from WASM.
    /// Useful for having an LLM stream strings/code logs instantly to the screen.
    pub fn canvas_draw_text(&self, target_id: &str, text: &str, x: f64, y: f64, font: &str, color: &str) -> Result<(), String> {
        if let Some(el) = self.document.get_element_by_id(target_id) {
            if let Ok(canvas) = el.dyn_into::<HtmlCanvasElement>() {
                if let Ok(Some(ctx_obj)) = canvas.get_context("2d") {
                    if let Ok(ctx) = ctx_obj.dyn_into::<CanvasRenderingContext2d>() {
                        ctx.set_font(font);
                        #[allow(deprecated)]
                        ctx.set_fill_style(&color.into());
                        let _ = ctx.fill_text(text, x, y);
                        return Ok(());
                    }
                }
            }
            Err(format!("Element '{}' is not a valid Canvas 2D", target_id))
        } else {
            Err(format!("Element with id '{}' not found", target_id))
        }
    }
}
