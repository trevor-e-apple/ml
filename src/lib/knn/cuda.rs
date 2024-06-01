fn add_test() {
    let lib: libloading::Library =
        unsafe { libloading::Library::new(dll_path).unwrap() };

    match init_cuda_device(&lib) {
        Ok(_) => {}
        Err(err) => {
            println!("init_cuda_device Error: {:?}", err);
            exit(1);
        }
    };

    let byte_count = 1024;
    let f32_count = byte_count / size_of::<f32>();
    let cuda_array_one = match alloc_cuda_mem(&lib, byte_count) {
        Ok(mem_ptr) => mem_ptr as *mut f32,
        Err(err) => {
            println!("alloc_cuda_mem Error {:?}", err);
            exit(1);
        }
    };
    let cuda_array_two = match alloc_cuda_mem(&lib, byte_count) {
        Ok(mem_ptr) => mem_ptr as *mut f32,
        Err(err) => {
            println!("alloc_cuda_mem Error {:?}", err);
            exit(1);
        }
    };
    let cuda_array_three = match alloc_cuda_mem(&lib, byte_count) {
        Ok(mem_ptr) => mem_ptr as *mut f32,
        Err(err) => {
            println!("alloc_cuda_mem Error {:?}", err);
            exit(1);
        }
    };

    println!("initializing host arrays");

    let mut host_mem_one = Vec::<f32>::with_capacity(f32_count);
    let mut host_mem_two = Vec::<f32>::with_capacity(f32_count);
    let mut host_mem_three = Vec::<f32>::with_capacity(f32_count);
    for index in 0..f32_count {
        host_mem_one.push(index as f32);
        host_mem_two.push(index as f32);
        host_mem_three.push(0.0);
    }

    println!("Transferring data to GPU arrays");
    match mem_to_device(
        &lib,
        cuda_array_one as *mut c_void,
        host_mem_one.as_ptr() as *const c_void,
        byte_count,
    ) {
        Ok(_) => println!("Data transferred"),
        Err(_) => todo!("Data failed to transfer"),
    };

    match mem_to_device(
        &lib,
        cuda_array_two as *mut c_void,
        host_mem_two.as_ptr() as *const c_void,
        byte_count,
    ) {
        Ok(_) => println!("Data transferred"),
        Err(_) => todo!("Data failed to transfer"),
    };

    println!("Adding arrays...");

    match cuda_add_arrays(
        &lib,
        cuda_array_one as *const f32,
        cuda_array_two as *const f32,
        f32_count,
        cuda_array_three as *mut f32,
    ) {
        Ok(_) => {
            println!("Added arrays...");
        }
        Err(_) => todo!("Something went wrong"),
    }

    println!("Mem back to host...");
    match mem_to_host(
        &lib,
        host_mem_three.as_mut_ptr() as *mut c_void,
        cuda_array_three as *const c_void,
        byte_count,
    ) {
        Ok(_) => {
            for value in host_mem_three {
                println!("{:?}", value);
            }
        }
        Err(_) => todo!("Something went wrong"),
    }
}
