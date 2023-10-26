mod lib;

use std::{
    format,
    slice,
    time::{SystemTime, UNIX_EPOCH},
    vec,
    path, println, fs::File, io::Read,
};

use metal::*;
use plotters::{
    prelude::{
        BitMapBackend, ChartBuilder, Circle, EmptyElement, IntoDrawingArea,
    },
    series::PointSeries,
    style::{IntoFont, BLUE, RED, WHITE},
};

use lib::{
    knn::{Label, Knn},
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


    let zero_zero_data = append_2d_data(
        rng,
        &mut data,
        &mut labels,
        &[0.0, 0.0],
        std_dev,
        samples_per_class,
        0,
    );
    let zero_one_data = append_2d_data(
        rng,
        &mut data,
        &mut labels,
        &[0.0, 1.0],
        std_dev,
        samples_per_class,
        1,
    );
    let one_zero_data = append_2d_data(
        rng,
        &mut data,
        &mut labels,
        &[1.0, 0.0],
        std_dev,
        samples_per_class,
        1,
    );
    let one_one_data = append_2d_data(
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

fn gen_random_data_mtl_buff(buff: &Buffer) {
    let pointer: *mut i32 = buff.contents() as *mut i32;
    let length = buff.length() as usize;
    let contents = unsafe { slice::from_raw_parts_mut(pointer, length) };
    for index in 0..length {
        contents[index] = 0;
    }
}

// TODO: generate some 2d spiral data

fn main() {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let mut rng = Rng::new(now.as_secs());

    let (training_data, training_labels) = gen_xor_data(&mut rng, 0.1, 90);
    let (test_data, _) = gen_xor_data(&mut rng, 0.15, 10);

    // predict test data classes
    let knn_predictor = Knn::new(5, training_data, training_labels);

    let (predicted_zero, predicted_one) = {
        let mut predicted_zero: Vec<(f32, f32)> = vec![];
        let mut predicted_one: Vec<(f32, f32)> = vec![];
        let predictions = knn_predictor.predict_classes(&test_data, 4);
        for (datum, predicted_label) in test_data.iter().zip(predictions.iter()) {
            let tuple_point = (datum[0], datum[1]);
            if *predicted_label == 0 {
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

    // metal test
    {
        let device = Device::system_default().unwrap();

        let library_path = path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/metal_kernels/add_array.metal");
        println!("{}", library_path.as_path().display().to_string());

        let mut kernel_file = File::open(library_path).unwrap();
        let mut contents = String::new();
        kernel_file.read_to_string(&mut contents).unwrap();

        let device = Device::system_default().expect("no device found");

        let options = CompileOptions::new();
        let _library = device.new_library_with_source(&contents, &options).unwrap();
        let function = _library.get_function("add_arrays", None).unwrap();
        let pipeline = device.new_compute_pipeline_state_with_function(&function);
        let command_queue = device.new_command_queue();
        let buffer_one = device.new_buffer(8192, MTLResourceOptions::empty());
        let buffer_two = device.new_buffer(8192, MTLResourceOptions::empty());
        let buffer_three = device.new_buffer(8192, MTLResourceOptions::empty());
    }
}
