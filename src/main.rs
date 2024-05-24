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

    // vulkan initialization

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
            Event::RedrawEventsCleared => {
                let image_extent: [u32; 2] = window.inner_size().into();
            }
            _ => ()
        }
    });
}
