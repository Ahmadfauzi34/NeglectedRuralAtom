use svg::Document;
use svg::node::element::{Circle, Rectangle};
use plotters::prelude::*;

/// A utility module exposing quick generic SVG and charting generators to the LLM Sandbox.
pub struct SvgGenerator;

impl SvgGenerator {
    /// Generates a simple SVG scatter plot representation of generic 2D points relative to a center.
    /// This is an example of direct XML/SVG construction for any abstract coordinate map.
    pub fn build_scatter_svg(center_x: f32, center_y: f32, points: &[(f32, f32)]) -> String {
        let mut document = Document::new()
            .set("viewBox", (0, 0, 200, 200))
            .set("width", "100%")
            .set("height", "100%")
            .add(
                Rectangle::new()
                    .set("x", 0)
                    .set("y", 0)
                    .set("width", 200)
                    .set("height", 200)
                    .set("fill", "#111")
            )
            .add(
                Circle::new()
                    .set("cx", 100)
                    .set("cy", 100)
                    .set("r", 90)
                    .set("stroke", "#4CAF50")
                    .set("stroke-width", 2)
                    .set("fill", "none")
            );

        // Draw the center anchor (origin)
        document = document.add(
            Circle::new()
                .set("cx", 100)
                .set("cy", 100)
                .set("r", 3)
                .set("fill", "#FFF")
        );

        for &(x, y) in points {
            // Map global coordinates relative to the center, scaled to fit the 200x200 box
            let rel_x = 100.0 + ((x - center_x) * 0.5);
            let rel_y = 100.0 + ((y - center_y) * 0.5);

            if rel_x >= 0.0 && rel_x <= 200.0 && rel_y >= 0.0 && rel_y <= 200.0 {
                document = document.add(
                    Circle::new()
                        .set("cx", rel_x)
                        .set("cy", rel_y)
                        .set("r", 2)
                        .set("fill", "#3b82f6") // Generic blue point
                );
            }
        }

        let mut out = Vec::new();
        svg::write(&mut out, &document).unwrap();
        String::from_utf8(out).unwrap_or_default()
    }

    /// Wraps standard HTML within an SVG `<foreignObject>`.
    /// This allows LLMs to embed complex HTML components (like forms, canvases, videos) inside SVG visualizations.
    pub fn build_foreign_object_svg(width: f32, height: f32, inner_html: &str) -> String {
        format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
  <foreignObject x="0" y="0" width="{width}" height="{height}">
    <div xmlns="http://www.w3.org/1999/xhtml" style="width: 100%; height: 100%;">
      {html}
    </div>
  </foreignObject>
</svg>"#,
            width = width,
            height = height,
            html = inner_html
        )
    }

    /// Uses the `plotters` library to draw a sophisticated line chart for any generic numeric series.
    /// Generates the raw SVG string that can be injected into the DOM.
    pub fn build_line_chart_svg(data_points: &[(f32, f32)], title: &str) -> String {
        let mut svg_buffer = String::new();

        {
            let root = SVGBackend::with_string(&mut svg_buffer, (400, 300)).into_drawing_area();
            root.fill(&WHITE).unwrap();

            // Find min/max for scaling
            let max_x = data_points.iter().map(|p| p.0).fold(0.0f32, f32::max);
            let max_y = data_points.iter().map(|p| p.1).fold(0.0f32, f32::max);

            let mut chart = ChartBuilder::on(&root)
                .caption(title, ("sans-serif", 20).into_font())
                .margin(5)
                .x_label_area_size(30)
                .y_label_area_size(30)
                .build_cartesian_2d(0f32..max_x, 0f32..max_y)
                .unwrap();

            chart.configure_mesh().draw().unwrap();

            chart.draw_series(LineSeries::new(
                data_points.iter().copied(),
                &RED,
            )).unwrap();

            root.present().unwrap();
        } // Drop the backend here so the string buffer unlocks

        svg_buffer
    }
}
