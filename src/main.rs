use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator,
        AutoCommandBufferBuilder,
        CommandBufferUsage,
        RenderPassBeginInfo,
        SubpassBeginInfo,
        SubpassContents
    },
    device::{
        physical::PhysicalDeviceType,
        Device,
        DeviceCreateInfo,
        DeviceExtensions,
        QueueCreateInfo,
        QueueFlags
    },
    image::{view::ImageView, Image, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState,
        GraphicsPipeline,
        PipelineLayout,
        PipelineShaderStageCreateInfo
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::{self, GpuFuture},
    Validated, VulkanError, VulkanLibrary
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder
};


fn main()
{
    let event_loop = EventLoop::new();

    // initialize vulkan and select a physical device

    let library = VulkanLibrary::new().unwrap();
    // get extensions need to draw to the window surface - "window-drawing" functions are not core;
    // therefore, they must be gotten and enabled manually
    let required_extensions = Surface::required_extensions(&event_loop);
    let instance = Instance::new(
        library,
        InstanceCreateInfo
        {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,  // allow for MoltenVK
            enabled_extensions: required_extensions,
            ..Default::default()
        }
    ).unwrap();


    // "Arc" provides a succinct way to count references to shared data - Arc is thread safe
    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    // surface defines where on the window can be drawn
    //
    // 'window.clone()' increments the reference count for the Arc
    let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();


    // extensions specify the capabilities needed by the program
    let device_extensions = DeviceExtensions
    {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };


    // choose the physical device to use
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p|
        {
            p.supported_extensions().contains(&device_extensions)
        })
        .filter_map(|p|
        {
            // queue families are used to execute draw commands
            //
            // multiple queues can be allocated to allow for parallelization
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)|
                {
                    // select a queue family which supports graphics operations
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                    // ensures selection of a queue family which can draw to the surface, as well
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)|
        {
            match p.properties().device_type
            {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5
            }
        })
        .expect("No suitable physical device found!");

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type
    );


    // initialize selected virtual device for interfacing with physical
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo
        {
            // features and extensions program needs
            enabled_extensions: device_extensions,
            // queues which will be used
            queue_create_infos: vec![QueueCreateInfo
            {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        }
    ).unwrap();

    // collect received queues
    let queue = queues.next().unwrap();


    // allocate swapchain and images
    //
    // swapchain holds the space where image data will be written
    let (mut swapchain, images) = {
        // get surface's capabilities for swapchain creation
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        // choose image format
        let image_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        Swapchain::new(
            device.clone(),
            surface,
            SwapchainCreateInfo
            {
                // minimum number of images required by the program - must have at least 2 for
                // fullscreen
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format,
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            }
        ).unwrap()
    };


    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));


    // allocate buffers for storing shape
    #[derive(BufferContents, Vertex)]
    #[repr(C)]  // force compiler to honor our chosen layout for vertices
    struct Vertex
    {
        #[format(R32G32_SFLOAT)]
        position: [f32; 2]
    }

    let vertices = [
        Vertex
        {
            position: [-0.5, -0.25]
        },
        Vertex
        {
            position: [0.0, 0.5]
        },
        Vertex
        {
            position: [0.25, -0.1]
        }
    ];
    let vertex_buffer = Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo
        {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo
        {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vertices
    ).unwrap();


    // create vertex and fragment shaders
    mod vs
    {
        vulkano_shaders::shader!
        {
            ty: "vertex",
            src: r"
                #version 450

                layout(location = 0) in vec2 position;

                void main()
                {
                    gl_Position = vec4(position, 0.0, 1.0);
                }
            "
        }
    }

    mod fs
    {
        vulkano_shaders::shader!
        {
            ty: "fragment",
            src: r"
                #version 450

                layout(location = 0) out vec4 f_color;

                void main()
                {
                    f_color = vec4(1.0, 0.0, 0.0, 1.0);
                }
            "
        }
    }


    // create render pass - defines image layout and where colors, depth and/or stencil info will
    // be written
    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments:
        {
            // 'color' is name of attachment
            color:
            {
                format: swapchain.image_format(),
                // 'samples' relates to the coloring of pixels, more samples will give a better
                // image
                samples: 1,  // increase to enable antialiasing
                // clear contents of attachment when drawing begins
                load_op: Clear,
                // store output of draw in 'actual' image
                store_op: Store
            }
        },
        pass:
        {
            color: [color],
            // no depth-stencil
            depth_stencil: {}
        }
    ).unwrap();


    // describes how an operation will be performed; in this case, how the program will produce its
    // graphics
    let pipeline = {
        // load shaders (i.e. vertex and fragment shaders defined above)
        let vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let vertex_input_state = Vertex::per_vertex()
            .definition(&vs.info().input_interface)
            .unwrap();

        // list of shader stages
        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs)
        ];

        // define pipeline layout - describes locations, types, and pushes constants
        //
        // many pipelines can share resources; therefore, it is more efficient to maximize the
        // resources which are shared between pipelines
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap()
        ).unwrap();

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo
            {
                stages: stages.into_iter().collect(),
                // defines how vertex data is read into the shader from the buffer
                vertex_input_state: Some(vertex_input_state),
                // defines arrangement of vertices into primitives - default is a triangle
                input_assembly_state: Some(InputAssemblyState::default()),
                // defines transforms and trimming to fit primities into the framebuffer
                viewport_state: Some(ViewportState::default()),
                // defines culling of polygons into pixel rasters - default does not cull
                rasterization_state: Some(RasterizationState::default()),
                // defines conversion of multiple fragment shaders samples into on pixel value
                multisample_state: Some(MultisampleState::default()),
                // defines the combination of existing pixel values with new ones
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default()
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            }
        ).unwrap()
    };


    let mut viewport = Viewport
    {
        offset: [0.0, 0.0],
        extent: [0.0, 0.0],
        depth_range: 0.0..=1.0
    };


    // allocate framebuffers

    // creates a different framebuffer for each image
    let mut framebuffers =
        window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);

    // create a command buffer allocator for managing the buffers
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());


    // end of vulkan initialization

    // for detecting window resize
    let mut recreate_swapchain = false;

    // store command submission of previous frame
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());


    event_loop.run(move |event, _, control_flow|
    {
        match event
        {
            Event::WindowEvent
            {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            },
            Event::WindowEvent
            {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::RedrawEventsCleared => {
            }
            _ => ()
        }
    });
}


fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport
) -> Vec<Arc<Framebuffer>>
{
    let extent = images[0].extent();
    viewport.extent = [extent[0] as f32, extent[1] as f32];

    images
        .iter()
        .map(|image|
        {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo
                {
                    attachments: vec![view],
                    ..Default::default()
                }
            ).unwrap()
        })
        .collect::<Vec<_>>()
}
