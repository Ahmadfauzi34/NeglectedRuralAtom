use svg::Document;
use svg::node::element::{Circle, Rectangle};
use plotters::prelude::*;

/// A utility module exposing quick SVG and charting generators to the LLM Sandbox.
pub struct SvgGenerator;

impl SvgGenerator {
    /// Generates a simple SVG radar or map representation of agents within a specific radius.
    /// This is a basic example of direct XML/SVG construction.
    pub fn build_radar_svg(center_x: f32, center_y: f32, agents: &[(f32, f32)]) -> String {
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

        // Draw the center anchor
        document = document.add(
            Circle::new()
                .set("cx", 100)
                .set("cy", 100)
                .set("r", 3)
                .set("fill", "#FFF")
        );

        for &(x, y) in agents {
            // Map global coordinates relative to the radar center, scaled to fit the 200x200 box
            let rel_x = 100.0 + ((x - center_x) * 0.5);
            let rel_y = 100.0 + ((y - center_y) * 0.5);

            if rel_x >= 0.0 && rel_x <= 200.0 && rel_y >= 0.0 && rel_y <= 200.0 {
                document = document.add(
                    Circle::new()
                        .set("cx", rel_x)
                        .set("cy", rel_y)
                        .set("r", 2)
                        .set("fill", "#F44336")
                );
            }
        }

        let mut out = Vec::new();
        svg::write(&mut out, &document).unwrap();
        String::from_utf8(out).unwrap_or_default()
    }

    /// Uses the `plotters` library to draw a sophisticated line chart (e.g. population metrics over time).
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
