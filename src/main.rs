use std::sync::Arc;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder
};


fn main()
{
    let event_loop = EventLoop::new();
    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());

    // vulkan initialization

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
