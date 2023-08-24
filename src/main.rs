mod lib;

use std::{
    format,
    time::{SystemTime, UNIX_EPOCH},
    vec,
};

use plotters::{
    prelude::{
        BitMapBackend, ChartBuilder, Circle, DrawingBackend, EmptyElement,
        IntoDrawingArea, Text,
    },
    series::{LineSeries, PointSeries},
    style::{IntoFont, RED, WHITE},
};

use lib::Rng;

fn main() {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let mut rng = Rng::new(now.as_secs());

    for _ in 0..100 {
        let random = rng.rand();
        println!("{random}");
    }

    // Create a 800*600 bitmap and start drawing
    let mut root =
        BitMapBackend::new("test/1.png", (300, 200)).into_drawing_area();
    root.fill(&WHITE);
    let mut chart_builder = ChartBuilder::on(&root);
    chart_builder
        .caption("This is our first plot", ("sans-serif", 40).into_font());
    chart_builder.x_label_area_size(20);
    chart_builder.y_label_area_size(40);
    let mut chart =
        chart_builder.build_cartesian_2d(0f32..10f32, 0f32..10f32).unwrap();

    chart
        .configure_mesh()
        .x_labels(5)
        .y_labels(5)
        .y_label_formatter(&|x| format!("{:.3}", x))
        .draw()
        .unwrap();

    // draw something in the drawing area
    chart
        .draw_series(LineSeries::new(
            vec![(0.0, 0.0), (5.0, 5.0), (8.0, 7.0)],
            &RED,
        ))
        .unwrap();

    // draw point series
    chart.draw_series(PointSeries::of_element(
        vec![(0.0, 0.0), (5.0, 5.0), (8.0, 7.0)],
        5,
        &RED,
        &|c, s, st| {
            return EmptyElement::at(c)
                + Circle::new((0, 0), s, st.filled())
                + Text::new(
                    format!("{:?}", c),
                    (10, 0),
                    ("sans-serif", 10).into_font(),
                );
        },
    )).unwrap();

    root.present().unwrap();
}
