use crate::{camera, geometry::{self, CompositeMesh, StaticMesh, Vertex}, texture, transforms, utils};
use anyhow::{anyhow, Result};
use bytemuck::{cast_slice, Pod, Zeroable};
use cgmath::{num_traits::clamp, *};
use std::{iter, path::PathBuf};
use std::collections::HashMap;
use std::mem::size_of;
use std::sync::Arc;
use wgpu::{BindGroup, BindGroupLayout, include_spirv_raw};
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};
use crate::primitives::{Color3f, Vec3f, Transform};
use crate::transforms::create_transforms;

static WINDOW_TITLE: &str = "Schedar Demo";

const IS_PERSPECTIVE: bool = true;

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
                    features: wgpu::Features::SPIRV_SHADER_PASSTHROUGH,
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
#[derive(Clone, Copy, Pod, Zeroable)]
struct DirectionalLight {
    direction: Vec3f,
    ambient: Color3f,
    diffuse: Color3f,
    specular: Color3f,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PointLight {
    position: Vec3f,
    ambient: Color3f,
    diffuse: Color3f,
    specular: Color3f,
    constant: f32,
    linear: f32,
    quadratic: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SpotLight {
    position: Vec3f,
    direction: Color3f,
    ambient: Color3f,
    diffuse: Color3f,
    specular: Color3f,
    cutoff: f32,
    outer_cutoff: f32,
    constant: f32,
    linear: f32,
    quadratic: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    model: cgmath::Matrix4<f32>,
    mvp: cgmath::Matrix4<f32>,
    normal: cgmath::Matrix4<f32>,
    eye_position: cgmath::Point3<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightsUniform {
    directional_light: DirectionalLight,
    // spot_light: SpotLight,
    point_light: PointLight,
    directional_enabled: u32,
    // spot_count: u32,
    point_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaterialParams {
    shininess: f32,
}

pub struct Actor {
    static_mesh: CompositeMesh,
    transform: Transform,
}

impl Actor {
    pub fn new(static_mesh: CompositeMesh, transform: Transform) -> Self {
        Self {
            static_mesh,
            transform,
        }
    }
}

pub struct DrawableActor {
    meshes: Vec<DrawableMesh>,
    textures: Vec<DrawableTexture>,
}

pub struct MeshPipeline {
    pipeline_layout: wgpu::PipelineLayout,
    pipeline: wgpu::RenderPipeline,
}

impl MeshPipeline {
    pub fn new(
        device: &wgpu::Device,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        lights_bind_group_layout: &wgpu::BindGroupLayout,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
        shader_vert: &wgpu::ShaderModule,
        shader_frag: &wgpu::ShaderModule,
        config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout, &lights_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_vert,
                entry_point: "main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_frag,
                entry_point: "main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
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
        Self {
            pipeline_layout,
            pipeline,
        }
    }
}

pub struct DrawableMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    indices_num: u32,
    material_id: usize,
}

impl DrawableMesh {
    pub fn new(
        device: &wgpu::Device,
        static_mesh: &StaticMesh,
        material_id: usize,
    ) -> DrawableMesh {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: cast_slice(&static_mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: cast_slice(&static_mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        DrawableMesh {
            vertex_buffer,
            index_buffer,
            material_id,
            indices_num: static_mesh.indices.len() as u32,
        }
    }

    // // TODO: how to fix?
    // pub fn render<'a, 'b: 'a>(
    //     &'b self,
    //     render_pass: &'a mut wgpu::RenderPass<'a>,
    //     camera_bind_group: &'a wgpu::BindGroup,
    //     texture_bind_group: &'a wgpu::BindGroup,
    // ) {
    //     render_pass.set_pipeline(&self.pipeline);
    //     render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
    //     render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    //     render_pass.set_bind_group(0, camera_bind_group, &[]);
    //     render_pass.set_bind_group(1, texture_bind_group, &[]);
    //     render_pass.draw_indexed(0..self.indices_num, 0, 0..1);
    // }
}

struct DrawableTexture {
    params_buffer: wgpu::Buffer,
    image_texture: texture::Texture,
    texture_bind_group: wgpu::BindGroup,
}

impl DrawableTexture {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
        material: &geometry::Material,
    ) -> Result<DrawableTexture> {
        let fallback = PathBuf::from("assets/brick.png");
        let path = if let Some(geometry::MaterialParam::Texture(path)) = &material.albedo {
            path
        } else {
            // return Err(anyhow!("non texture encountered for albedo"));
            &fallback
        };
        let image_texture = texture::Texture::create_texture_data(
            device,
            queue,
            path.as_path(),
            wgpu::AddressMode::Repeat,
            wgpu::AddressMode::Repeat,
        )?;
        let material_params_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Material Params Uniform Buffer"),
            size: size_of::<MaterialParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let material_params = MaterialParams {
            shininess: 76.8,
        };
        queue.write_buffer(
            &material_params_uniform_buffer,
            0,
            bytemuck::cast_slice(&[material_params]),
        );

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: material_params_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&image_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&image_texture.sampler),
                },
            ],
            label: Some("Texture Bind Group"),
        });
        let result = DrawableTexture {
            params_buffer: material_params_uniform_buffer,
            image_texture,
            texture_bind_group,
        };
        Ok(result)
    }
}

struct State {
    pub init: InitWgpu,

    // scene entities
    drawable_actors: HashMap<Arc<str>, DrawableActor>,
    // shaders: HashMap<Arc<str>, wgpu::ShaderModule>,
    shader_vert: wgpu::ShaderModule,
    shader_frag: wgpu::ShaderModule,
    pipeline: MeshPipeline,
    texture_bind_group_layout: BindGroupLayout,

    // camera
    camera: camera::Camera,
    projection: Matrix4<f32>,
    camera_controller: camera::CameraController,
    camera_uniform_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    mouse_pressed: bool,

    // light
    lights_uniform_buffer: wgpu::Buffer,
    lights_bind_group: BindGroup,
    light_position: Vec3f,
    going_backwards: bool,

    // smaa
    smaa_target: smaa::SmaaTarget,
}

impl State {
    async fn new(window: &Window) -> Result<Self> {
        let init = InitWgpu::init_wgpu(window).await?;

        // fixme: strange problem with passing of vector into shader - they're broken

        let camera = camera::Camera::new((0.0, 10.0, 0.0), cgmath::Deg(-20.0), cgmath::Deg(0.0));
        let camera_controller = camera::CameraController::new(0.3, 30.0);
        let aspect = init.config.width as f32 / init.config.height as f32;
        let projection = transforms::create_projection(aspect, IS_PERSPECTIVE);

        // stores model and mvp matrix
        let camera_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniform Buffer"),
            size: size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // stores light_position and eye_position
        let lights_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lights Uniform Buffer"),
            size: size_of::<LightsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // let light_uniform_buffer = init.device.create_buffer(&wgpu::BufferDescriptor {
        //     label: Some("Light Uniform Buffer"),
        //     size: 48,
        //     usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        //     mapped_at_creation: false,
        // });

        // init.queue.write_buffer(
        //     &light_uniform_buffer,
        //     0,
        //     bytemuck::cast_slice(&[light_data]),
        // );

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
                    ],
                    label: Some("Camera Bind Group Layout"),
                });

        let camera_bind_group = init.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("Camera Bind Group"),
        });

        let lights_bind_group_layout = init.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("Lights Bind Group Layout"),
        });

        let lights_bind_group = init.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &lights_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lights_uniform_buffer.as_entire_binding(),
                },
            ],
            label: Some("Lights Bind Group"),
        });

        let texture_bind_group_layout =
            init.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
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
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("Texture Bind Group Layout"),
            });

        let (shader_vert, shader_frag) = utils::load_spirv_shader(&init.device, "todo");

        let pipeline = MeshPipeline::new(&init.device, &camera_bind_group_layout, &lights_bind_group_layout, &texture_bind_group_layout, &shader_vert, &shader_frag, &init.config);

        let mut smaa_target = smaa::SmaaTarget::new(
            &init.device,
            &init.queue,
            window.inner_size().width,
            window.inner_size().height,
            init.config.format,
            smaa::SmaaMode::Smaa1X,
        );

        let result = Self {
            init,
            drawable_actors: HashMap::new(),
            shader_vert,
            shader_frag,
            pipeline,
            texture_bind_group_layout,
            camera,
            projection,
            camera_controller,
            camera_bind_group,
            camera_uniform_buffer,
            lights_uniform_buffer,
            lights_bind_group,
            smaa_target,
            mouse_pressed: false,
            light_position: Vec3f::new(0.0, 10.0, 0.0),
            going_backwards: false,
        };
        Ok(result)
    }

    pub fn spawn_actor(&mut self, name: Arc<str>, actor: &Actor, shader_name: Arc<str>) -> Result<()> {
        if self.drawable_actors.contains_key(&name) {
            return Err(anyhow!("asset with given name already spawned"));
        }
        let mut meshes = Vec::new();
        let mut textures = Vec::new();
        for (static_mesh, mat_id) in actor.static_mesh.components.iter() {
            if let Some(mat_id) = *mat_id {
                let mesh = DrawableMesh::new(
                    &self.init.device,
                    static_mesh,
                    mat_id,
                );
                meshes.push(mesh);
            } else {
                println!("encountered mesh without referenced texture");
            }
        }
        for mat in actor.static_mesh.materials.iter() {
            let tex = DrawableTexture::new(
                &self.init.device,
                &self.init.queue,
                &self.texture_bind_group_layout,
                mat,
            )?;
            textures.push(tex)
        }
        let actor = DrawableActor {
            meshes,
            textures,
        };
        self.drawable_actors.insert(name, actor);
        Ok(())
    }

    // pub fn add_shader(&mut self, name: Arc<str>, _path: Arc<str>) -> Result<()> {
    //
    //     self.shaders.insert(name.clone(), shader);
    //     Ok(())
    // }

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
            self.smaa_target
                .resize(&self.init.device, new_size.width, new_size.height);
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
        if self.light_position.x > 120.0 {
            // self.light_position.x = -150.0;
            self.going_backwards = true;
        }
        if self.light_position.x < -120.0 {
            // self.light_position.x = -150.0;
            self.going_backwards = false;
        }
        if self.going_backwards {
            self.light_position.x -= dt * 50.0;
        } else {
            self.light_position.x += dt * 50.0;
        }

        // update camera uniform
        let model_mat =
            transforms::create_transforms([0.0, 0.0, 0.0].into(), [0.0, 0.0, 0.0].into(), [0.1, 0.1, 0.1].into());
        self.camera_controller.update_camera(&mut self.camera, dt);
        let normal_mat = model_mat
            .invert()
            .ok_or_else(|| anyhow!("matrix is not invertible"))
            .unwrap()
            .transpose();
        let camera_uniform = CameraUniform {
            model: model_mat,
            mvp: self.projection * self.camera.view_mat() * model_mat,
            normal: normal_mat,
            eye_position: self.camera.position,
        };
        self.init
            .queue
            .write_buffer(&self.camera_uniform_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));

        // update lights uniform
        let directional_light = DirectionalLight {
            direction: Vec3f::new(1.0, 0.0, 0.0),
            ambient: Color3f::new(1.0, 1.0, 1.0),
            diffuse: Color3f::new(1.0, 1.0, 1.0),
            specular: Color3f::new(1.0, 1.0, 1.0),
        };
        let point_light = PointLight {
            position: self.light_position,
            ambient: Color3f::new(1.0, 1.0, 1.0),
            diffuse: Color3f::new(1.0, 1.0, 1.0),
            specular: Color3f::new(1.0, 1.0, 1.0),
            constant: 1.0,
            linear: 0.09,
            quadratic: 0.032,
        };
        let light_data = LightsUniform {
            directional_light,
            point_light,
            directional_enabled: 1,
            point_count: 1,
        };
        self.init.queue.write_buffer(
            &self.lights_uniform_buffer,
            0,
            bytemuck::cast_slice(&[light_data]),
        )
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.init.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let smaa_frame = self
            .smaa_target
            .start_frame(&self.init.device, &self.init.queue, &view);

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
                    view: &(*smaa_frame),
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

            render_pass.set_pipeline(&self.pipeline.pipeline);

            for (_, actor) in self.drawable_actors.iter() {
                for mesh in actor.meshes.iter() {
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                    render_pass.set_bind_group(1, &self.lights_bind_group, &[]);
                    render_pass.set_bind_group(
                        2,
                        &actor.textures[mesh.material_id].texture_bind_group,
                        &[],
                    );
                    render_pass.draw_indexed(0..mesh.indices_num, 0, 0..1);
                }
            }
        }

        self.init.queue.submit(iter::once(encoder.finish()));

        smaa_frame.resolve();
        output.present();

        Ok(())
    }
}

pub fn run(actors: &[(Arc<str>, &Actor)]) -> Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop)?;
    window.set_title(WINDOW_TITLE);
    let mut state = pollster::block_on(State::new(&window))?;

    for (name, actor) in actors {
        state.spawn_actor(name.clone(), actor, "todo".into())?;
    }

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
