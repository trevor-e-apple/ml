mod lib;

use std::{
    format,
    time::{SystemTime, UNIX_EPOCH},
    vec,
};

use plotters::{
    prelude::{
        BitMapBackend, ChartBuilder, Circle, EmptyElement, IntoDrawingArea,
    },
    series::PointSeries,
    style::{IntoFont, BLUE, RED, WHITE},
};

use lib::{Rng, box_muller};

fn gen_xor_data(rng: &mut Rng, mu: &[f32; 2], std_dev: f32, samples: usize) -> Vec<(f32, f32)> {
    let mut result: Vec<(f32, f32)> = Vec::with_capacity(samples);

    for _ in 0..samples {
        let x = box_muller(rng, mu[0] as f64, std_dev as f64);
        let y = box_muller(rng, mu[1] as f64, std_dev as f64);

        result.push((x as f32, y as f32));
    }

    result
}

fn main() {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let mut rng = Rng::new(now.as_secs());

    for _ in 0..100 {
        let random = rng.rand();
        println!("{random}");
    }

    for i in 0..5 {
        let sample = box_muller(&mut rng, 5.0, 2.5);
        println!("sample {i}: {sample}");
    }

    let zero_zero_data = gen_xor_data(&mut rng, &[0.0, 0.0], 0.1, 100);
    let zero_one_data = gen_xor_data(&mut rng, &[0.0, 1.0], 0.1, 100);
    let one_zero_data = gen_xor_data(&mut rng, &[1.0, 0.0], 0.1, 100);
    let one_one_data = gen_xor_data(&mut rng, &[1.0, 1.0], 0.1, 100);


    // Create a 800*600 bitmap and start drawing
    let root =
        BitMapBackend::new("test/xor.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart_builder = ChartBuilder::on(&root);
    chart_builder.caption("XOR", ("sans-serif", 40).into_font());
    chart_builder.x_label_area_size(20);
    chart_builder.y_label_area_size(40);
    let mut chart =
        chart_builder.build_cartesian_2d(-1f32..2f32, -1f32..2f32).unwrap();

    chart
        .configure_mesh()
        .x_labels(5)
        .y_labels(5)
        .y_label_formatter(&|x| format!("{:.3}", x))
        .draw()
        .unwrap();

    // draw point series
    chart
        .draw_series(PointSeries::of_element(
            zero_zero_data,
            5,
            &RED,
            &|c, s, st| {
                return EmptyElement::at(c)
                    + Circle::new((0, 0), s, st.filled());
            },
        ))
        .unwrap();

    chart
        .draw_series(PointSeries::of_element(
            zero_one_data,
            5,
            &BLUE,
            &|c, s, st| {
                return EmptyElement::at(c)
                    + Circle::new((0, 0), s, st.filled());
            },
        ))
        .unwrap();

    chart
        .draw_series(PointSeries::of_element(
            one_zero_data,
            5,
            &BLUE,
            &|c, s, st| {
                return EmptyElement::at(c)
                    + Circle::new((0, 0), s, st.filled());
            },
        ))
        .unwrap();

    chart
        .draw_series(PointSeries::of_element(
            one_one_data,
            5,
            &RED,
            &|c, s, st| {
                return EmptyElement::at(c)
                    + Circle::new((0, 0), s, st.filled());
            },
        ))
        .unwrap();

    root.present().unwrap();
}
