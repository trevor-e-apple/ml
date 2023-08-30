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

use lib::{
    knn::{Class, Knn},
    rng::{box_muller, Rng},
};

fn append_2d_data(
    rng: &mut Rng,
    data: &mut Vec<(Class, Vec<f32>)>,
    mu: &[f32; 2],
    std_dev: f32,
    samples: usize,
    class: Class,
) {
    for _ in 0..samples {
        let x = box_muller(rng, mu[0] as f64, std_dev as f64);
        let y = box_muller(rng, mu[1] as f64, std_dev as f64);

        data.push((class, vec![x as f32, y as f32]));
    }
}

fn gen_xor_data(
    rng: &mut Rng, std_dev: f32, samples_per_class: usize
) -> Vec<(Class, Vec<f32>)> {
    let mut result: Vec<(Class, Vec<f32>)> = vec![];
    
    let zero_zero_data = append_2d_data(
        rng,
        &mut result,
        &[0.0, 0.0],
        std_dev,
        samples_per_class,
        0,
    );
    let zero_one_data = append_2d_data(
        rng,
        &mut result,
        &[0.0, 1.0],
        std_dev,
        samples_per_class,
        1,
    );
    let one_zero_data = append_2d_data(
        rng,
        &mut result,
        &[1.0, 0.0],
        std_dev,
        samples_per_class,
        1,
    );
    let one_one_data = append_2d_data(
        rng,
        &mut result,
        &[1.0, 1.0],
        std_dev,
        samples_per_class,
        0,
    );

    result
}

// TODO: generate some 2d spiral data

fn main() {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let mut rng = Rng::new(now.as_secs());

    let training_data = gen_xor_data(&mut rng, 0.1, 90);
    let test_data = gen_xor_data(&mut rng, 0.15, 10);

    // predict test data classes
    let knn_predictor = Knn::new(5, training_data);

    let (predicted_zero, predicted_one) = {
        let mut predicted_zero: Vec<(f32, f32)> = vec![];
        let mut predicted_one: Vec<(f32, f32)> = vec![];
        for datum in test_data {
            let predicted_class = knn_predictor.predict_class(&datum.1);

            let tuple_point = (datum.1[0], datum.1[1]);
            if predicted_class == 0 {
                predicted_zero.push(tuple_point);
            } else {
                predicted_one.push(tuple_point);
            }
        }

        (predicted_zero, predicted_one)
    };

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
            predicted_zero,
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
            predicted_one,
            5,
            &BLUE,
            &|c, s, st| {
                return EmptyElement::at(c)
                    + Circle::new((0, 0), s, st.filled());
            },
        ))
        .unwrap();

    root.present().unwrap();
}
