use metal::*;

fn gen_data_mtl_buff(buff: &Buffer) {
    let pointer: *mut f32 = buff.contents() as *mut f32;
    let length = buff.length() as usize;
    let contents = unsafe { slice::from_raw_parts_mut(pointer, length) };
    for index in 0..length {
        contents[index] = index as f32;
    }
}

fn print_mtl_buff(buff: &Buffer) {
    let pointer: *mut f32 = buff.contents() as *mut f32;
    let length = buff.length() as usize;
    let contents = unsafe { slice::from_raw_parts_mut(pointer, length) };
    println!("{:?}", contents);
}

fn metal_test() {
    let library_path = path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/metal_kernels/add_array.metal");
    println!("{}", library_path.as_path().display().to_string());

    let mut kernel_file = File::open(library_path).unwrap();
    let mut contents = String::new();
    kernel_file.read_to_string(&mut contents).unwrap();

    let device = Device::system_default().expect("no device found");

    let options = CompileOptions::new();
    let _library = device.new_library_with_source(&contents, &options).unwrap();
    let function = _library.get_function("add_arrays", None).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&function).unwrap();
    let command_queue = device.new_command_queue();
    let array_length: u64 = 8192;
    let mut buffer_one =
        device.new_buffer(array_length, MTLResourceOptions::empty());
    let mut buffer_two =
        device.new_buffer(array_length, MTLResourceOptions::empty());
    let mut buffer_three =
        device.new_buffer(array_length, MTLResourceOptions::empty());

    gen_data_mtl_buff(&mut buffer_one);
    gen_data_mtl_buff(&mut buffer_two);
    gen_data_mtl_buff(&mut buffer_three);

    let command_buffer = command_queue.new_command_buffer();
    let compute_encoder = command_buffer.new_compute_command_encoder();

    compute_encoder.set_compute_pipeline_state(&pipeline);
    compute_encoder.set_buffer(0, Some(&buffer_one), 0);
    compute_encoder.set_buffer(1, Some(&buffer_two), 0);
    compute_encoder.set_buffer(2, Some(&buffer_three), 0);
    let grid_size = MTLSize { width: array_length, height: 1, depth: 1 };
    let thread_group_len = pipeline.max_total_threads_per_threadgroup();
    let thread_group_len = if thread_group_len > array_length {
        array_length
    } else {
        thread_group_len
    };
    let thread_group_size =
        MTLSize { width: thread_group_len, height: 1, depth: 1 };
    compute_encoder.dispatch_threads(grid_size, thread_group_size);
    compute_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // print contents of result buffer
    print_mtl_buff(&buffer_three);
}
