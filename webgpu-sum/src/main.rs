    use log::info;
    use wgpu::util::DeviceExt;

fn main() {
    // Initialize the logger
    env_logger::init();

    // Run the async function synchronously
    pollster::block_on(run());
}

async fn run() {
    // Create a new instance
    let instance = wgpu::Instance::default();

    // Request an adapter
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to find a suitable GPU adapter");

    // Request a device and queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Main Device"),
                required_features: wgpu::Features::empty(), 
                required_limits: wgpu::Limits::default(),  
                memory_hints: wgpu::MemoryHints::default(), 
                trace: wgpu::Trace::default(),
            }
        )
        .await
        .expect("Failed to create device");

    // Log adapter information
    info!("Using adapter: {:?}", adapter.get_info());
    println!("Device and queue successfully initialized!");

    // Define the Input Data
    let input_data: Vec<u32> = (1..=1024).collect();

    // Size of the buffer in bytes
    let input_buffer_size = (input_data.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
    let number_workgroups = input_data.len() / 64;
    let result_buffer_size = (number_workgroups * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

    // Create the input buffer
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(&input_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Create the output buffer
    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Result Buffer"),
        size: result_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    println!("Input and Result buffers created!");

    // Include the shader
    let shader_src = include_str!("shaders/sum_reduction.wgsl");
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Sum Reduction Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    println!("Shader included!");

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            // inputBuffer binding
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // resultBuffer binding
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    println!("Bind Group Layout defined!");

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            // inputBuffer
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            // resultBuffer
            wgpu::BindGroupEntry {
                binding: 1,
                resource: result_buffer.as_entire_binding(),
            },
        ],
    });
    println!("Bind Group defined!");

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    println!("Pipeline Layout defined!");

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    println!("Compute Pipeline defined!");

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Compute Encoder"),
    });
    println!("Encoder added!");
    
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(number_workgroups as u32, 1, 1); // Adjust workgroup count as needed
    }
    println!("Began the compute pass, set the pipeline, bind group, and dispatched the shader!");
    
    let staging_buffer = std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: result_buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    println!("Created Staging Buffer!");

    // Copy Data From the Result Buffer
    encoder.copy_buffer_to_buffer(&result_buffer, 0, &staging_buffer, 0, result_buffer_size);
    println!("Copied Data From the Result Buffer to Staging Buffer!");

    // Submit commands
    queue.submit(Some(encoder.finish()));
    println!("Submitted command buffer!");

    // Ensure all GPU work has completed before mapping
    match device.poll(wgpu::MaintainBase::Wait) {
        Ok(_) =>  println!("GPU commands completed!"),
        Err(_) => panic!("GPU commands failed!")
    }

    // Map the buffer and wait for completion
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();

    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });

    pollster::block_on(async {
        loop {
            match device.poll(wgpu::MaintainBase::Poll) {
                Ok(_) =>  println!("GPU Poll completed!"),
                Err(_) => panic!("GPU Poll failed!")
            }
            if let Some(result) = receiver.receive().await {
                result.expect("Failed to map buffer");
                break;
            }
        }
    });

    // Explicitly scoped mapped range
    {
        let mapped_data = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice::<u8, u32>(&mapped_data);
        println!("Read mapped data: {:?}", result);
        println!("Final result (after CPU aggregation): {:?}", result.iter().sum::<u32>());
    } // mapped_data is dropped here explicitly

    staging_buffer.unmap();
}