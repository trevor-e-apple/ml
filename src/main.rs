mod lib;

use std::{
    format,
    iter::zip,
    time::{SystemTime, UNIX_EPOCH},
    vec,
};

use plotters::{
    prelude::{
        BitMapBackend, ChartBuilder, Circle, EmptyElement, IntoDrawingArea,
    },
    series::PointSeries,
    style::{Color, IntoFont, RGBColor, BLUE, RED, WHITE},
};

use lib::{
    knn::{Knn, Label},
    rng::{box_muller, Rng},
};

fn append_2d_data(
    rng: &mut Rng,
    data: &mut Vec<Vec<f32>>,
    labels: &mut Vec<Label>,
    mu: &[f32; 2],
    std_dev: f32,
    samples: usize,
    label: Label,
) {
    for _ in 0..samples {
        let x = box_muller(rng, mu[0] as f64, std_dev as f64);
        let y = box_muller(rng, mu[1] as f64, std_dev as f64);

        data.push(vec![x as f32, y as f32]);
        labels.push(label);
    }
}

fn gen_xor_data(
    rng: &mut Rng,
    std_dev: f32,
    samples_per_class: usize,
) -> (Vec<Vec<f32>>, Vec<Label>) {
    let mut data: Vec<Vec<f32>> = vec![];
    let mut labels: Vec<Label> = Vec::with_capacity(4 * samples_per_class);

    append_2d_data(
        rng,
        &mut data,
        &mut labels,
        &[0.0, 0.0],
        std_dev,
        samples_per_class,
        0,
    );
    append_2d_data(
        rng,
        &mut data,
        &mut labels,
        &[0.0, 1.0],
        std_dev,
        samples_per_class,
        1,
    );
    append_2d_data(
        rng,
        &mut data,
        &mut labels,
        &[1.0, 0.0],
        std_dev,
        samples_per_class,
        1,
    );
    append_2d_data(
        rng,
        &mut data,
        &mut labels,
        &[1.0, 1.0],
        std_dev,
        samples_per_class,
        0,
    );

    (data, labels)
}

fn gen_2d_spiral_data(
    rng: &mut Rng,
    std_dev: f32,
    class_count: usize,
    samples_per_class: usize,
) -> (Vec<(f32, f32)>, Vec<Label>) {
    let mut data: Vec<(f32, f32)> = vec![];
    let mut labels: Vec<Label> =
        Vec::with_capacity(class_count * samples_per_class);

    for class_index in 0..class_count {
        let angle_offset =
            2.0 * 3.14 * (class_index as f64 / class_count as f64);
        for sample_index in 0..samples_per_class {
            // radius shall be between 0 and 1
            let radius: f64 =
                (sample_index as f64) / (samples_per_class as f64);
            let angle =
                2.0 * 3.14 * (sample_index as f64 / samples_per_class as f64)
                    + angle_offset;
            // offset so we have different spirals
            let (true_y, true_x) = angle.sin_cos();
            // let (x, y) = (radius * true_x, radius * true_y);
            let x = box_muller(rng, radius * true_x, std_dev as f64);
            let y = box_muller(rng, radius * true_y, std_dev as f64);

            data.push((x as f32, y as f32));
            labels.push(class_index as Label);
        }
    }

    (data, labels)
}

fn draw_2d_data(
    file_path: &str,
    caption: &str,
    all_point_series: Vec<(Vec<(f32, f32)>, &RGBColor)>,
) {
    // Create a 800*600 bitmap and start drawing
    let root = BitMapBackend::new(file_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart_builder = ChartBuilder::on(&root);
    chart_builder.caption(caption, ("sans-serif", 40).into_font());
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
    for (point_series, color) in all_point_series {
        chart
            .draw_series(PointSeries::of_element(
                point_series,
                5,
                &color,
                &|c, s, st| {
                    return EmptyElement::at(c)
                        + Circle::new((0, 0), s, st.filled());
                },
            ))
            .unwrap();
    }

    root.present().unwrap();
}

fn main() {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let mut rng = Rng::new(now.as_secs());

    let (training_data, training_labels) = gen_xor_data(&mut rng, 0.1, 90);
    let (test_data, _) = gen_xor_data(&mut rng, 0.15, 10);

    let (spiral_training_data, spiral_training_labels) =
        gen_2d_spiral_data(&mut rng, 0.02, 2, 100);

    // predict test data classes
    let knn_predictor = Knn::new(5, training_data, training_labels);

    let (predicted_zero, predicted_one) = {
        let mut predicted_zero: Vec<(f32, f32)> = vec![];
        let mut predicted_one: Vec<(f32, f32)> = vec![];
        let predictions = knn_predictor.predict_classes(&test_data, 4);
        for (datum, predicted_label) in test_data.iter().zip(predictions.iter())
        {
            let tuple_point = (datum[0], datum[1]);
            if *predicted_label == 0 {
                predicted_zero.push(tuple_point);
            } else {
                predicted_one.push(tuple_point);
            }
        }

        (predicted_zero, predicted_one)
    };

    draw_2d_data(
        "test/xor.png",
        "XOR",
        vec![(predicted_zero, &RED), (predicted_one, &BLUE)],
    );

    let red_spiral = {
        let mut red_spiral: Vec<(f32, f32)> = vec![];
        for (data, label) in zip(&spiral_training_data, &spiral_training_labels)
        {
            if *label == 0 {
                red_spiral.push(*data);
            }
        }
        red_spiral
    };

    let blue_spiral = {
        let mut blue_spiral: Vec<(f32, f32)> = vec![];
        for (data, label) in zip(&spiral_training_data, &spiral_training_labels)
        {
            if *label == 1 {
                blue_spiral.push(*data);
            }
        }
        blue_spiral
    };
    draw_2d_data(
        "test/spiral.png",
        "SPIRAL",
        vec![(red_spiral, &RED), (blue_spiral, &BLUE)],
    );
}
