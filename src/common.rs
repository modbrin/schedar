use crate::{
    camera,
    geometry::{StaticMesh, Vertex},
    texture, transforms,
};
use anyhow::{anyhow, Result};
use bytemuck::{cast_slice, Pod, Zeroable};
use cgmath::{num_traits::clamp, *};
use std::iter;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

static WINDOW_TITLE: &str = "Schedar Demo";

pub struct InitWgpu {
    pub surface: wgpu::Surface,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
}

impl InitWgpu {
    pub async fn init_wgpu(window: &Window) -> Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
        });
        let surface = unsafe { instance.create_surface(window)? };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow!("failed to select appropriate adapter"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None, // api call tracing
            )
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(&surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: *surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
        })
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Light {
    specular_color: [f32; 4],
    ambient_intensity: f32,
    diffuse_intensity: f32,
    specular_intensity: f32,
    specular_shininess: f32,
}

pub fn light(sc: [f32; 3], ai: f32, di: f32, si: f32, ss: f32) -> Light {
    Light {
        specular_color: [sc[0], sc[1], sc[2], 1.0],
        ambient_intensity: ai,
        diffuse_intensity: di,
        specular_intensity: si,
        specular_shininess: ss,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_project_mat: [[f32; 4]; 4],
}

const IS_PERSPECTIVE: bool = true;

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_project_mat: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_project(&mut self, camera: &camera::Camera, project_mat: Matrix4<f32>) {
        self.view_project_mat = (project_mat * camera.view_mat()).into()
    }
}

struct State {
    pub init: InitWgpu,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    indices_num: u32,

    camera: camera::Camera,
    projection: Matrix4<f32>,
    camera_controller: camera::CameraController,
    camera_uniform: CameraUniform,
    camera_bind_group: wgpu::BindGroup,
    vs_uniform_buffer: wgpu::Buffer,
    fs_uniform_buffer: wgpu::Buffer,
    _light_uniform_buffer: wgpu::Buffer,
    mouse_pressed: bool,

    _image_texture: texture::Texture,
    texture_bind_group: wgpu::BindGroup,
    light_position: Point3<f32>,
}

impl State {
    async fn new(
        window: &Window,
        static_mesh: &StaticMesh,
        light_data: Light,
        img_file: &str,
        u_mode: wgpu::AddressMode,
        v_mode: wgpu::AddressMode,
    ) -> Result<Self> {
        let init = InitWgpu::init_wgpu(window).await?;

        let image_texture = texture::Texture::create_texture_data(
            &init.device,
            &init.queue,
            img_file,
            u_mode,
            v_mode,
        )?;
        let texture_bind_group_layout =
            init.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                    label: Some("Texture Bind Group Layout"),
                });

        let texture_bind_group = init.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&image_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&image_texture.sampler),
                },
            ],
            label: Some("Texture Bind Group"),
        });

        let shader = init
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            });

        let camera = camera::Camera::new((3.0, 1.5, 3.0), cgmath::Deg(-80.0), cgmath::Deg(-30.0));
        let camera_controller = camera::CameraController::new(0.3, 5.0);
        let aspect = init.config.width as f32 / init.config.height as f32;
        let projection = transforms::create_projection(aspect, IS_PERSPECTIVE);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_project(&camera, projection);

        // stores model_mal and view_projection_mat
        let vs_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Uniform Buffer"),
            size: 192,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // stores light_position and eye_position
        let fs_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fragment Uniform Buffer"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let light_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Uniform Buffer"),
            size: 48,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        init.queue.write_buffer(
            &light_uniform_buffer,
            0,
            bytemuck::cast_slice(&[light_data]),
        );

        let camera_bind_group_layout =
            init.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                    label: Some("Camera Bind Group Layout"),
                });

        let camera_bind_group = init.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vs_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: fs_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: light_uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("Camera Bind Group"),
        });

        let pipeline_layout = init
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = init
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[Vertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: init.config.format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent::REPLACE,
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24Plus,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        let vertex_buffer = init
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: cast_slice(&static_mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let index_buffer = init
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: cast_slice(&static_mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        let result = Self {
            init,
            pipeline,
            vertex_buffer,
            index_buffer,
            indices_num: static_mesh.indices.len() as u32,
            camera,
            projection,
            camera_controller,
            camera_bind_group,
            camera_uniform,
            vs_uniform_buffer,
            fs_uniform_buffer,
            _light_uniform_buffer: light_uniform_buffer,
            mouse_pressed: false,
            _image_texture: image_texture,
            texture_bind_group,
            light_position: Point3::new(0.0, 0.0, 2.0),
        };
        Ok(result)
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            let aspect = new_size.width as f32 / new_size.height as f32;
            self.projection = transforms::create_projection(aspect, IS_PERSPECTIVE);

            self.init.size = new_size;
            self.init.config.width = new_size.width;
            self.init.config.height = new_size.height;
            self.init
                .surface
                .configure(&self.init.device, &self.init.config);
        }
    }

    #[allow(unused_variables)]
    fn input(&mut self, event: &DeviceEvent) -> bool {
        match event {
            DeviceEvent::Button {
                button: 3, // Right Mouse Button
                state,
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                true
            }
            DeviceEvent::MouseMotion { delta } => {
                if self.mouse_pressed {
                    self.camera_controller.mouse_move(delta.0, delta.1);
                }
                true
            }
            _ => false,
        }
    }

    fn update(&mut self, dt: f32) {
        // update light
        if self.light_position.x > 3.0 {
            self.light_position.x = -3.0;
        }
        self.light_position.x += dt;

        // camera update
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform
            .update_view_project(&self.camera, self.projection);

        // model + normal
        let model_mat =
            transforms::create_transforms([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let normal_mat = model_mat
            .invert()
            .ok_or_else(|| anyhow!("matrix is not invertible"))
            .unwrap()
            .transpose();
        let model_ref: &[f32; 16] = model_mat.as_ref();
        let normal_ref: &[f32; 16] = normal_mat.as_ref();

        // write vert shader data
        self.init
            .queue
            .write_buffer(&self.vs_uniform_buffer, 0, bytemuck::cast_slice(model_ref));
        self.init.queue.write_buffer(
            &self.vs_uniform_buffer,
            64,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
        self.init.queue.write_buffer(
            &self.vs_uniform_buffer,
            128,
            bytemuck::cast_slice(normal_ref),
        );

        // write frag shader data
        let light_position: &[f32; 3] = self.light_position.as_ref();
        let eye_position: &[f32; 3] = self.camera.position.as_ref();
        self.init.queue.write_buffer(
            &self.fs_uniform_buffer,
            0,
            bytemuck::cast_slice(light_position),
        );
        self.init.queue.write_buffer(
            &self.fs_uniform_buffer,
            16,
            bytemuck::cast_slice(eye_position),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.init.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = self.init.device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: self.init.config.width,
                height: self.init.config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: None,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.init
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.2,
                            g: 0.247,
                            b: 0.314,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: false,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.texture_bind_group, &[]);
            render_pass.draw_indexed(0..self.indices_num, 0, 0..1);
        }

        self.init.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub fn run(
    vertex_data: &StaticMesh,
    light_data: Light,
    file_name: &str,
    u_mode: wgpu::AddressMode,
    v_mode: wgpu::AddressMode,
) -> Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop)?;
    window.set_title(WINDOW_TITLE);
    let mut state = pollster::block_on(State::new(
        &window,
        &vertex_data,
        light_data,
        &file_name,
        u_mode,
        v_mode,
    ))?;

    let mut render_start_time = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| match event {
        Event::DeviceEvent { ref event, .. } => {
            state.input(event);
        }
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        ..
                    },
                ..
            } => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(physical_size) => {
                state.resize(*physical_size);
            }
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                state.resize(**new_inner_size);
            }
            WindowEvent::KeyboardInput {
                input:
                    winit::event::KeyboardInput {
                        virtual_keycode: Some(keycode),
                        state: key_state,
                        ..
                    },
                is_synthetic: false,
                ..
            } => {
                let offset = (key_state == &ElementState::Pressed)
                    .then_some(1.0)
                    .unwrap_or(-1.0);
                let mut input = state.camera_controller.get_movement_input();
                match keycode {
                    VirtualKeyCode::D => input.x = clamp(input.x + offset, -1.0, 1.0),
                    VirtualKeyCode::A => input.x = clamp(input.x - offset, -1.0, 1.0),
                    VirtualKeyCode::W => input.y = clamp(input.y + offset, -1.0, 1.0),
                    VirtualKeyCode::S => input.y = clamp(input.y - offset, -1.0, 1.0),
                    VirtualKeyCode::E => input.z = clamp(input.z + offset, -1.0, 1.0),
                    VirtualKeyCode::Q => input.z = clamp(input.z - offset, -1.0, 1.0),
                    _ => (),
                }
                state.camera_controller.set_movement_input(input);
            }
            _ => {}
        },
        Event::RedrawRequested(_) => {
            let now = std::time::Instant::now();
            let dt = (now - render_start_time).as_secs_f32();
            render_start_time = now;
            state.update(dt);

            match state.render() {
                Ok(_) => {}
                Err(wgpu::SurfaceError::Lost) => state.resize(state.init.size),
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        _ => {}
    });
}
