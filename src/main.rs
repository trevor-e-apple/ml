mod lib;

use std::{
    env, format,
    iter::zip,
    mem::size_of,
    os::raw::c_void,
    process::exit,
    ptr,
    time::{SystemTime, UNIX_EPOCH},
    vec,
};

use plotters::{
    prelude::{
        BitMapBackend, ChartBuilder, Circle, EmptyElement, IntoDrawingArea,
    },
    series::PointSeries,
    style::{
        IntoFont, RGBColor, BLUE, CYAN, GREEN, MAGENTA, RED, WHITE, YELLOW,
    },
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
) -> (Vec<Vec<f32>>, Vec<Label>) {
    let mut data: Vec<Vec<f32>> = vec![];
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

            data.push(vec![x as f32, y as f32]);
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

fn init_cuda_device(
    dll_path: &str,
) -> Result<c_void, Box<dyn std::error::Error>> {
    unsafe {
        let lib = libloading::Library::new(dll_path)?;
        let func: libloading::Symbol<unsafe extern "C" fn() -> c_void> =
            lib.get(b"?init_cuda_device@@YAXXZ")?;
        Ok(func())
    }
}

fn alloc_cuda_mem(
    dll_path: &str,
    byte_count: usize,
) -> Result<*mut c_void, Box<dyn std::error::Error>> {
    unsafe {
        let lib = libloading::Library::new(dll_path)?;
        let func: libloading::Symbol<
            unsafe extern "C" fn(usize) -> *mut c_void,
        > = lib.get(b"?alloc_cuda_mem@@YAPEAX_K@Z")?;
        Ok(func(byte_count))
    }
}

fn cuda_add_arrays(
    dll_path: &str,
    a: *const f32,
    b: *const f32,
    len: usize,
    out: *mut f32,
) -> Result<c_void, Box<dyn std::error::Error>> {
    unsafe {
        let lib = libloading::Library::new(dll_path)?;
        let func: libloading::Symbol<
            unsafe extern "C" fn(
                *const f32,
                *const f32,
                usize,
                *mut f32,
            ) -> c_void,
        > = lib.get(b"?add@@YAXPEAM0_K0@Z")?;
        Ok(func(a, b, len, out))
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let dll_path = match args.get(1) {
        Some(path) => path,
        None => {
            println!("Unable to find dll path");
            exit(1);
        }
    };
    println!("Using dll at: {}", dll_path);

    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let mut rng = Rng::new(now.as_secs());

    let (training_data, training_labels) = gen_xor_data(&mut rng, 0.1, 90);
    let (test_data, _) = gen_xor_data(&mut rng, 0.15, 10);

    let (spiral_training_data, spiral_training_labels) =
        gen_2d_spiral_data(&mut rng, 0.02, 3, 100);
    let (spiral_test_data, spiral_test_labels) =
        gen_2d_spiral_data(&mut rng, 0.04, 3, 100);

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

    let (red_spiral, blue_spiral, green_spiral): (
        Vec<(f32, f32)>,
        Vec<(f32, f32)>,
        Vec<(f32, f32)>,
    ) = {
        let mut red_spiral: Vec<(f32, f32)> = vec![];
        let mut blue_spiral: Vec<(f32, f32)> = vec![];
        let mut green_spiral: Vec<(f32, f32)> = vec![];

        for (data, label) in zip(&spiral_training_data, &spiral_training_labels)
        {
            let tuple_point = (data[0], data[1]);
            if *label == 0 {
                red_spiral.push(tuple_point);
            } else if *label == 1 {
                blue_spiral.push(tuple_point);
            } else if *label == 2 {
                green_spiral.push(tuple_point);
            }
        }
        (red_spiral, blue_spiral, green_spiral)
    };

    let spiral_predictor =
        Knn::new(5, spiral_training_data, spiral_training_labels);

    let (
        red_spiral_correct,
        red_spiral_wrong,
        blue_spiral_correct,
        blue_spiral_wrong,
        green_spiral_correct,
        green_spiral_wrong,
    ) = {
        let mut red_spiral_correct: Vec<(f32, f32)> = vec![];
        let mut red_spiral_wrong: Vec<(f32, f32)> = vec![];
        let mut blue_spiral_correct: Vec<(f32, f32)> = vec![];
        let mut blue_spiral_wrong: Vec<(f32, f32)> = vec![];
        let mut green_spiral_correct: Vec<(f32, f32)> = vec![];
        let mut green_spiral_wrong: Vec<(f32, f32)> = vec![];

        let predictions =
            spiral_predictor.predict_classes(&spiral_test_data, 4);
        for ((datum, predicted_label), actual_label) in spiral_test_data
            .iter()
            .zip(predictions.iter())
            .zip(spiral_test_labels.iter())
        {
            let (correct_vec, wrong_vec) = if *actual_label == 0 {
                (&mut red_spiral_correct, &mut red_spiral_wrong)
            } else if *actual_label == 1 {
                (&mut blue_spiral_correct, &mut blue_spiral_wrong)
            } else if *actual_label == 2 {
                (&mut green_spiral_correct, &mut green_spiral_wrong)
            } else {
                panic!("something went horribly wrong");
            };

            let tuple_point = (datum[0], datum[1]);
            if *predicted_label == *actual_label {
                correct_vec.push(tuple_point)
            } else {
                wrong_vec.push(tuple_point);
            }
        }

        (
            red_spiral_correct,
            red_spiral_wrong,
            blue_spiral_correct,
            blue_spiral_wrong,
            green_spiral_correct,
            green_spiral_wrong,
        )
    };

    draw_2d_data(
        "test/spiral.png",
        "SPIRAL",
        vec![(red_spiral, &RED), (blue_spiral, &BLUE), (green_spiral, &GREEN)],
    );
    draw_2d_data(
        "test/spiral_knn_test.png",
        "SPIRAL",
        vec![
            (red_spiral_correct, &RED),
            (red_spiral_wrong, &MAGENTA),
            (blue_spiral_correct, &BLUE),
            (blue_spiral_wrong, &CYAN),
            (green_spiral_correct, &GREEN),
            (green_spiral_wrong, &YELLOW),
        ],
    );

    match init_cuda_device(dll_path) {
        Ok(_) => {}
        Err(err) => {
            println!("init_cuda_device Error: {:?}", err);
            exit(1);
        }
    };

    let byte_count = 1024;
    let f32_count = byte_count / size_of::<f32>();
    let cuda_array_one = match alloc_cuda_mem(dll_path, byte_count) {
        Ok(mem_ptr) => {
            ptr::slice_from_raw_parts_mut(mem_ptr as *mut f32, f32_count)
        }
        Err(err) => {
            println!("alloc_cuda_mem Error {:?}", err);
            exit(1);
        }
    };
    let cuda_array_two = match alloc_cuda_mem(dll_path, byte_count) {
        Ok(mem_ptr) => {
            ptr::slice_from_raw_parts_mut(mem_ptr as *mut f32, f32_count)
        }
        Err(err) => {
            println!("alloc_cuda_mem Error {:?}", err);
            exit(1);
        }
    };
    let cuda_array_three = match alloc_cuda_mem(dll_path, byte_count) {
        Ok(mem_ptr) => {
            ptr::slice_from_raw_parts_mut(mem_ptr as *mut f32, f32_count)
        }
        Err(err) => {
            println!("alloc_cuda_mem Error {:?}", err);
            exit(1);
        }
    };

    for index in 0..byte_count {
        unsafe {
            (*cuda_array_one)[index] = index as f32;
            (*cuda_array_two)[index] = index as f32;
        }
    }

    match cuda_add_arrays(
        dll_path,
        cuda_array_one as *const f32,
        cuda_array_two as *const f32,
        f32_count,
        cuda_array_three as *mut f32,
    ) {
        Ok(_) => todo!(),
        Err(_) => todo!(),
    }
}
