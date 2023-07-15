use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::{cmp, iter};

use anyhow::{anyhow, Result};
use bytemuck::cast_slice;
use encase::ShaderType;
use glam::*;
use num::clamp;
use wgpu::util::DeviceExt;
use wgpu::{BlendFactor, BlendOperation};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use crate::primitives::{ShaderTypeDefaultExt, ShaderTypeExt, Transform};
use crate::utils::*;
use crate::{
    camera,
    geometry::{self, CompositeMesh, MaterialParam, StaticMesh, Vertex},
    texture, transforms, utils,
};

static WINDOW_TITLE: &str = "Schedar Demo";

const BACKGROUND_CLR_COLOR: Vec4 = Vec4::new(0.2, 0.247, 0.314, 1.0);
const IS_PERSPECTIVE: bool = true;

// Spot lights count
const SPOT_LIGHTS_PER_PASS: usize = 16;
const MAX_SPOT_LIGHTS_PER_SCENE: usize = 500;
const SPOT_LIGHT_UNIFORMS_PER_SCENE: usize = MAX_SPOT_LIGHTS_PER_SCENE / SPOT_LIGHTS_PER_PASS;
const SPOT_LIGHTS_PER_SCENE: usize = SPOT_LIGHT_UNIFORMS_PER_SCENE * SPOT_LIGHTS_PER_PASS;

// Point lights count
const POINT_LIGHTS_PER_PASS: usize = 16;
const MAX_POINT_LIGHTS_PER_SCENE: usize = 1000;
const POINT_LIGHT_UNIFORMS_PER_SCENE: usize = MAX_POINT_LIGHTS_PER_SCENE / POINT_LIGHTS_PER_PASS;
const POINT_LIGHTS_PER_SCENE: usize = POINT_LIGHT_UNIFORMS_PER_SCENE * POINT_LIGHTS_PER_PASS;

pub struct WgpuContext {
    pub surface: wgpu::Surface,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
}

impl WgpuContext {
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
        let limits = wgpu::Limits {
            max_push_constant_size: 128,
            ..Default::default()
        };
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::SPIRV_SHADER_PASSTHROUGH
                        | wgpu::Features::PUSH_CONSTANTS,
                    limits,
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

#[derive(Clone, Copy, ShaderType)]
struct DirectionalLight {
    direction: Vec3,
    ambient: Vec3,
    diffuse: Vec3,
    specular: Vec3,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: Vec3::new(1.0, 0.0, 0.0),
            ambient: Vec3::new(1.0, 1.0, 1.0),
            diffuse: Vec3::new(1.0, 1.0, 1.0),
            specular: Vec3::new(1.0, 1.0, 1.0),
        }
    }
}

#[derive(Clone, Copy, ShaderType)]
struct PointLight {
    position: Vec3,
    ambient: Vec3,
    diffuse: Vec3,
    specular: Vec3,
    constant: f32,
    linear: f32,
    quadratic: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 0.0),
            ambient: Vec3::new(1.0, 1.0, 1.0),
            diffuse: Vec3::new(1.0, 1.0, 1.0),
            specular: Vec3::new(1.0, 1.0, 1.0),
            constant: 1.0,
            linear: 0.09,
            quadratic: 0.032,
        }
    }
}

#[derive(Clone, Copy, Default, ShaderType)]
struct SpotLight {
    position: Vec3,
    direction: Vec3,
    ambient: Vec3,
    diffuse: Vec3,
    specular: Vec3,
    cutoff: f32,
    outer_cutoff: f32,
    constant: f32,
    linear: f32,
    quadratic: f32,
}

#[derive(Clone, Copy, Default, ShaderType)]
struct SplitLightsBaseUniform {
    directional_light: DirectionalLight,
    directional_enabled: u32,
}

#[derive(Clone, Copy, Default, ShaderType)]
struct SplitLightsAddUniform {
    // spot_lights: [SpotLight, SPOT_LIGHTS_PER_PASS],
    point_lights: [PointLight; POINT_LIGHTS_PER_PASS],
    // spot_count: u32,
    point_count: u32,
}

impl SplitLightsAddUniform {
    /// Initializes internal buffer from slice, if slice is larger than buffer size then
    /// only fitting elements are used
    pub fn from_slice(point_lights: impl AsRef<[PointLight]>) -> Self {
        let slice = point_lights.as_ref();
        let len = cmp::min(slice.len(), POINT_LIGHTS_PER_PASS);
        let mut data = [PointLight::default(); POINT_LIGHTS_PER_PASS];
        data[..len].copy_from_slice(&slice[..len]);
        Self {
            point_lights: data,
            point_count: len as u32,
        }
    }
}

#[derive(Clone, Copy, Default, ShaderType)]
struct CameraUniform {
    view_project: Mat4,
    eye_position: Vec3,
}

#[derive(Clone, Copy, Default, ShaderType)]
struct ModelPushConst {
    model_mat: Mat4,
    normal_mat: Mat4,
}

impl ModelPushConst {
    pub fn from_transform(transform: &Transform) -> Self {
        let model_mat = transforms::create_transforms(
            transform.get_position(),
            transform.get_rotation(),
            transform.get_scale(),
        );
        let normal_mat = model_mat.inverse().transpose();
        Self {
            model_mat,
            normal_mat,
        }
    }
}

#[derive(Clone, Copy, Default, ShaderType)]
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
    transform: Transform,
}

pub struct DrawPipeline {
    pipeline_layout: wgpu::PipelineLayout,
    pipeline: wgpu::RenderPipeline,
}

impl DrawPipeline {
    pub fn new(
        device: &wgpu::Device,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        lights_bind_group_layout: &wgpu::BindGroupLayout,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
        shader_vert: &wgpu::ShaderModule,
        shader_frag: &wgpu::ShaderModule,
        config: &wgpu::SurfaceConfiguration,
        is_additive: bool,
    ) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &camera_bind_group_layout,
                &lights_bind_group_layout,
                &texture_bind_group_layout,
            ],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::VERTEX,
                range: 0..ModelPushConst::default_size() as u32,
            }],
        });
        let color_target_state = if is_additive {
            wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: BlendFactor::One, // Raw addition for lights layers onto base
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent::OVER, // TODO: is this correct?
                }),
                write_mask: wgpu::ColorWrites::ALL,
            }
        } else {
            wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent::REPLACE,
                    alpha: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            }
        };
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
                targets: &[Some(color_target_state)],
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

    // TODO: how to use this?
    // pub fn render<'a, 'b: 'a>(
    //     pipeline: &'b wgpu::RenderPipeline,
    //     vertex_buffer: &'b wgpu::Buffer,
    //     index_buffer: &'b wgpu::Buffer,
    //     indices_num: u32,
    //     render_pass: &'a mut wgpu::RenderPass<'a>,
    //     camera_bind_group: &'a wgpu::BindGroup,
    //     texture_bind_group: &'a wgpu::BindGroup,
    // ) {
    //     render_pass.set_pipeline(&pipeline);
    //     render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
    //     render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    //     render_pass.set_bind_group(0, camera_bind_group, &[]);
    //     render_pass.set_bind_group(1, texture_bind_group, &[]);
    //     render_pass.draw_indexed(0..indices_num, 0, 0..1);
    // }
}

#[derive(Default)]
struct DrawableImageTextures {
    albedo: Option<texture::Texture>,
    normal: Option<texture::Texture>,
    specular: Option<texture::Texture>,
    emission: Option<texture::Texture>,
}

struct DrawableTexture {
    params_buffer: wgpu::Buffer,
    // image_texture: texture::Texture,
    texture_bind_group: wgpu::BindGroup,
}

impl DrawableTexture {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
        material: &geometry::Material,
    ) -> Result<DrawableTexture> {
        let mut bind_entries = Vec::new();
        let mut material_params = MaterialParams { shininess: 76.8 };
        if let Some(MaterialParam::Scalar(s)) = material.shininess {
            material_params.shininess = s; // TODO: use texture
        }
        let material_params_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Material Params Uniform Buffer"),
            size: material_params.size().get(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // push to front to reserve space
        bind_entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: material_params_uniform_buffer.as_entire_binding(),
        });
        let mut image_textures = DrawableImageTextures::default();
        Self::prepare_bind_entries(
            device,
            queue,
            &mut bind_entries,
            &mut image_textures.albedo,
            material.albedo.as_ref(),
            1,
            geometry::DIFFUSE_TEX_FALLBACK_COLOR,
        )?;
        Self::prepare_bind_entries(
            device,
            queue,
            &mut bind_entries,
            &mut image_textures.specular,
            material.specular.as_ref(),
            5,
            geometry::SPECULAR_TEX_FALLBACK_COLOR,
        )?;
        Self::prepare_bind_entries(
            device,
            queue,
            &mut bind_entries,
            &mut image_textures.normal,
            material.normal.as_ref(),
            9,
            geometry::NORMAL_TEX_FALLBACK_COLOR,
        )?;
        Self::prepare_bind_entries(
            device,
            queue,
            &mut bind_entries,
            &mut image_textures.emission,
            material.emission.as_ref(),
            13,
            geometry::EMISSION_TEX_FALLBACK_COLOR,
        )?;
        queue.write_buffer(
            &material_params_uniform_buffer,
            0,
            &material_params.as_uniform_buffer_bytes(),
        );
        bind_entries[0] = wgpu::BindGroupEntry {
            binding: 0,
            resource: material_params_uniform_buffer.as_entire_binding(),
        };
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &bind_entries,
            label: Some("Texture Bind Group"),
        });
        let result = DrawableTexture {
            params_buffer: material_params_uniform_buffer,
            texture_bind_group,
        };
        Ok(result)
    }

    fn prepare_bind_entries<'a, 'b: 'a>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_entries: &'a mut Vec<wgpu::BindGroupEntry<'b>>,
        image_texture_tmp: &'b mut Option<texture::Texture>,
        mat_param: Option<&MaterialParam>,
        bind_num: u32,
        fallback_color: Vec3,
    ) -> Result<()> {
        if let Some(mparam) = mat_param {
            let image_texture = match mparam {
                MaterialParam::Texture(path) => texture::Texture::create_texture_data(
                    device,
                    queue,
                    path,
                    wgpu::AddressMode::Repeat,
                    wgpu::AddressMode::Repeat,
                )?,
                MaterialParam::Color(col) => {
                    texture::Texture::create_from_color(device, queue, *col)?
                }
                MaterialParam::Scalar(s) => {
                    let col = Vec3::new(*s, *s, *s);
                    texture::Texture::create_from_color(device, queue, col)?
                }
            };
            *image_texture_tmp = Some(image_texture);
        } else {
            let image_texture = texture::Texture::create_from_color(device, queue, fallback_color)?;
            *image_texture_tmp = Some(image_texture);
        }
        bind_entries.push(wgpu::BindGroupEntry {
            binding: bind_num,
            resource: wgpu::BindingResource::TextureView(&image_texture_tmp.as_ref().unwrap().view),
        });
        bind_entries.push(wgpu::BindGroupEntry {
            binding: bind_num + 1,
            resource: wgpu::BindingResource::Sampler(&image_texture_tmp.as_ref().unwrap().sampler),
        });
        Ok(())
    }
}

struct State {
    ctx: WgpuContext,
    scene_state: SceneState,
    render_state: RenderState,
    camera_state: CameraState,
    lights_state: LightsState,
}

struct SceneState {
    drawable_actors: HashMap<String, DrawableActor>,
}

struct RenderState {
    pipeline_base: DrawPipeline,
    pipeline_additive: DrawPipeline,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    smaa_target: smaa::SmaaTarget,
}

struct CameraState {
    camera: camera::Camera,
    projection: Mat4,
    camera_controller: camera::CameraController,
    camera_uniform_buffer: wgpu::Buffer,
    camera_bind_group_layout: wgpu::BindGroupLayout,
    camera_bind_group: wgpu::BindGroup,
    mouse_pressed: bool,
}

struct LightsState {
    directional_light: DirectionalLight,
    point_lights: Vec<PointLight>,
    spot_lights: Vec<SpotLight>,
    lights_base_uniform_buffer: wgpu::Buffer,
    lights_base_bind_group_layout: wgpu::BindGroupLayout,
    lights_base_bind_group: wgpu::BindGroup,
    // lights_add_uniform_buffer: wgpu::Buffer,
    // lights_add_bind_group: wgpu::BindGroup,
    lights_update_buffer: wgpu::Buffer,
    lights_update_buffer_bind_group_layout: wgpu::BindGroupLayout,
    lights_update_buffer_bind_group: wgpu::BindGroup,
    lights_update_buffer_elem_size: u64,
}

impl State {
    async fn new(window: &Window) -> Result<Self> {
        let ctx = WgpuContext::init_wgpu(window).await?;
        let scene_state = Self::init_scene_state();
        let camera_state = Self::init_camera_state(&ctx);
        let lights_state = Self::init_lights_state(&ctx);
        let render_state = Self::init_render_state(&ctx, window, &camera_state, &lights_state)?;

        let state = Self {
            ctx,
            scene_state,
            camera_state,
            render_state,
            lights_state,
        };
        Ok(state)
    }

    pub fn spawn_actor(&mut self, name: &str, actor: &Actor, shader_name: Arc<str>) -> Result<()> {
        if self.scene_state.drawable_actors.contains_key(name) {
            return Err(anyhow!("asset with given name already spawned"));
        }
        let mut meshes = Vec::new();
        let mut textures = Vec::new();
        for (static_mesh, mat_id) in actor.static_mesh.components.iter() {
            if let Some(mat_id) = *mat_id {
                let mesh = DrawableMesh::new(&self.ctx.device, static_mesh, mat_id);
                meshes.push(mesh);
            } else {
                println!("encountered mesh without referenced texture");
            }
        }
        for mat in actor.static_mesh.materials.iter() {
            let tex = DrawableTexture::new(
                &self.ctx.device,
                &self.ctx.queue,
                &self.render_state.texture_bind_group_layout,
                mat,
            )?;
            textures.push(tex)
        }
        let actor = DrawableActor {
            meshes,
            textures,
            transform: actor.transform.clone(),
        };
        self.scene_state
            .drawable_actors
            .insert(name.to_string(), actor);
        Ok(())
    }

    pub fn add_point_light(&mut self, position: Vec3, color: Vec3) {
        let light = PointLight {
            position,
            ambient: color,
            diffuse: color,
            ..Default::default()
        };
        self.lights_state.point_lights.push(light);
    }

    pub fn init_scene_state() -> SceneState {
        SceneState {
            drawable_actors: HashMap::new(),
        }
    }

    pub fn init_render_state(
        ctx: &WgpuContext,
        window: &Window,
        camera_state: &CameraState,
        lights_state: &LightsState,
    ) -> Result<RenderState> {
        let texture_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        // material params
                        make_common_bgl_entry_uniform(0),
                        // diffuse1 texture
                        make_common_bgl_entry_texture(1),
                        // diffuse1 sampler
                        make_common_bgl_entry_sampler(2),
                        // specular1 texture
                        make_common_bgl_entry_texture(5),
                        // specular1 sampler
                        make_common_bgl_entry_sampler(6),
                        // normal1 texture
                        make_common_bgl_entry_texture(9),
                        // normal1 sampler
                        make_common_bgl_entry_sampler(10),
                        // emissive1 texture
                        make_common_bgl_entry_texture(13),
                        // emissive1 sampler
                        make_common_bgl_entry_sampler(14),
                    ],
                    label: Some("Texture Bind Group Layout"),
                });
        let shader_vert = utils::load_spirv_shader_module(
            &ctx.device,
            "base_vert",
            "./shaders/out/light_vert.spv",
        )?;
        let shader_base_frag = utils::load_spirv_shader_module(
            &ctx.device,
            "base_frag",
            "./shaders/out/light_split_base_frag.spv",
        )?;
        let shader_add_frag = utils::load_spirv_shader_module(
            &ctx.device,
            "add_frag",
            "./shaders/out/light_split_add_frag.spv",
        )?;
        let pipeline_base = DrawPipeline::new(
            &ctx.device,
            &camera_state.camera_bind_group_layout,
            &lights_state.lights_base_bind_group_layout,
            &texture_bind_group_layout,
            &shader_vert,
            &shader_base_frag,
            &ctx.config,
            false,
        );
        let pipeline_additive = DrawPipeline::new(
            &ctx.device,
            &camera_state.camera_bind_group_layout,
            &lights_state.lights_update_buffer_bind_group_layout,
            &texture_bind_group_layout,
            &shader_vert,
            &shader_add_frag,
            &ctx.config,
            true,
        );
        let smaa_target = Self::init_smma_target(&ctx, window);
        let state = RenderState {
            pipeline_base,
            pipeline_additive,
            texture_bind_group_layout,
            smaa_target,
        };
        Ok(state)
    }

    pub fn init_smma_target(ctx: &WgpuContext, window: &Window) -> smaa::SmaaTarget {
        smaa::SmaaTarget::new(
            &ctx.device,
            &ctx.queue,
            window.inner_size().width,
            window.inner_size().height,
            ctx.config.format,
            smaa::SmaaMode::Smaa1X,
        )
    }

    pub fn init_camera_state(ctx: &WgpuContext) -> CameraState {
        let camera =
            camera::Camera::new((0.0, 10.0, 0.0), -20.0f32.to_radians(), 0.0f32.to_radians());
        let camera_controller = camera::CameraController::new(0.3, 30.0);
        let aspect = ctx.config.width as f32 / ctx.config.height as f32;
        let projection = transforms::create_projection(aspect, IS_PERSPECTIVE);

        // stores model and mvp matrix
        let camera_uniform_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniform Buffer"),
            size: CameraUniform::default_size(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                    label: Some("Camera Bind Group Layout"),
                });
        let camera_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_uniform_buffer.as_entire_binding(),
            }],
            label: Some("Camera Bind Group"),
        });
        CameraState {
            camera,
            projection,
            camera_controller,
            camera_uniform_buffer,
            camera_bind_group_layout,
            camera_bind_group,
            mouse_pressed: false,
        }
    }

    pub fn init_lights_state(ctx: &WgpuContext) -> LightsState {
        let lights_base_uniform_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lights Uniform Buffer"),
            size: SplitLightsBaseUniform::default_size(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights_base_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                    label: Some("Lights Bind Group Layout"),
                });
        let lights_base_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &lights_base_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lights_base_uniform_buffer.as_entire_binding(),
            }],
            label: Some("Lights Base Bind Group"),
        });
        let lights_update_buffer_elem_size = round_to_next_multiple(
            SplitLightsAddUniform::default_size(),
            ctx.device.limits().min_uniform_buffer_offset_alignment as u64,
        );
        let lights_update_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lights Update Buffer"),
            size: lights_update_buffer_elem_size * POINT_LIGHT_UNIFORMS_PER_SCENE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights_update_buffer_bind_group_layout = ctx.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: NonZeroU64::new(SplitLightsAddUniform::default_size()),
                    },
                    count: None,
                }],
                label: Some("Lights Update Buffer Bind Group Layout"),
            },
        );
        let lights_update_buffer_bind_group =
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &lights_update_buffer_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &lights_update_buffer,
                        offset: 0,
                        size: NonZeroU64::new(SplitLightsAddUniform::default_size()),
                    }),
                }],
                label: Some("Lights Ring Buffer Bind Group"),
            });
        LightsState {
            directional_light: DirectionalLight::default(),
            point_lights: Vec::new(),
            spot_lights: Vec::new(),
            lights_base_uniform_buffer,
            lights_base_bind_group_layout,
            lights_base_bind_group,
            lights_update_buffer,
            lights_update_buffer_bind_group_layout,
            lights_update_buffer_bind_group,
            lights_update_buffer_elem_size,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            let aspect = new_size.width as f32 / new_size.height as f32;
            self.camera_state.projection = transforms::create_projection(aspect, IS_PERSPECTIVE);

            self.ctx.size = new_size;
            self.ctx.config.width = new_size.width;
            self.ctx.config.height = new_size.height;
            self.ctx
                .surface
                .configure(&self.ctx.device, &self.ctx.config);
            self.render_state
                .smaa_target
                .resize(&self.ctx.device, new_size.width, new_size.height);
        }
    }

    #[allow(unused_variables)]
    fn input(&mut self, event: &DeviceEvent) -> bool {
        match event {
            DeviceEvent::Button {
                button: 3, // Right Mouse Button
                state,
            } => {
                self.camera_state.mouse_pressed = *state == ElementState::Pressed;
                true
            }
            DeviceEvent::MouseMotion { delta } => {
                if self.camera_state.mouse_pressed {
                    self.camera_state
                        .camera_controller
                        .mouse_move(delta.0, delta.1);
                }
                true
            }
            _ => false,
        }
    }

    fn update(&mut self, dt: f32) {
        // update camera
        self.camera_state
            .camera_controller
            .update_camera(&mut self.camera_state.camera, dt);
        let camera_uniform = CameraUniform {
            view_project: self.camera_state.projection * self.camera_state.camera.view_mat(),
            eye_position: self.camera_state.camera.position,
        };
        self.ctx.queue.write_buffer(
            &self.camera_state.camera_uniform_buffer,
            0,
            &camera_uniform.as_uniform_buffer_bytes(),
        );

        // update lights uniform
        let light_base_data = SplitLightsBaseUniform {
            directional_light: self.lights_state.directional_light.clone(),
            directional_enabled: 1,
        };
        self.ctx.queue.write_buffer(
            &self.lights_state.lights_base_uniform_buffer,
            0,
            &light_base_data.as_uniform_buffer_bytes(),
        )
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.ctx.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let smaa_frame =
            self.render_state
                .smaa_target
                .start_frame(&self.ctx.device, &self.ctx.queue, &view);
        let depth_texture = self.ctx.device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: self.ctx.config.width,
                height: self.ctx.config.height,
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
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        {
            let background_color = BACKGROUND_CLR_COLOR;
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Base Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &(*smaa_frame),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: background_color.x as f64,
                            g: background_color.y as f64,
                            b: background_color.z as f64,
                            a: background_color.w as f64,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_pipeline(&self.render_state.pipeline_base.pipeline);

            for (_, actor) in self.scene_state.drawable_actors.iter() {
                let model_push_const = ModelPushConst::from_transform(&actor.transform);
                let pconst_data = model_push_const.as_uniform_buffer_bytes();
                render_pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, &pconst_data);
                render_pass.set_bind_group(0, &self.camera_state.camera_bind_group, &[]);
                render_pass.set_bind_group(1, &self.lights_state.lights_base_bind_group, &[]);
                for mesh in actor.meshes.iter() {
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.set_bind_group(
                        2,
                        &actor.textures[mesh.material_id].texture_bind_group,
                        &[],
                    );
                    render_pass.draw_indexed(0..mesh.indices_num, 0, 0..1);
                }
            }
        }
        let mut lights_buf_idx = 0u64;
        for lights_chunk in self.lights_state.point_lights.chunks(POINT_LIGHTS_PER_PASS) {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Additive Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &(*smaa_frame),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });
            render_pass.set_pipeline(&self.render_state.pipeline_additive.pipeline);

            let light_add_data = SplitLightsAddUniform::from_slice(lights_chunk);
            self.ctx.queue.write_buffer(
                &self.lights_state.lights_update_buffer,
                lights_buf_idx,
                &light_add_data.as_uniform_buffer_bytes(),
            );
            render_pass.set_bind_group(
                1,
                &self.lights_state.lights_update_buffer_bind_group,
                &[lights_buf_idx as u32],
            );
            lights_buf_idx += self.lights_state.lights_update_buffer_elem_size;

            for (_, actor) in self.scene_state.drawable_actors.iter() {
                let model_push_const = ModelPushConst::from_transform(&actor.transform);
                let pconst_data = model_push_const.as_uniform_buffer_bytes();
                render_pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, &pconst_data);
                render_pass.set_bind_group(0, &self.camera_state.camera_bind_group, &[]);
                for mesh in actor.meshes.iter() {
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.set_bind_group(
                        2,
                        &actor.textures[mesh.material_id].texture_bind_group,
                        &[],
                    );
                    render_pass.draw_indexed(0..mesh.indices_num, 0, 0..1);
                }
            }
        }
        self.ctx.queue.submit(iter::once(encoder.finish()));

        smaa_frame.resolve();
        output.present();

        Ok(())
    }
}

pub fn run(actors: &[(&str, &Actor)]) -> Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop)?;
    window.set_title(WINDOW_TITLE);
    let mut state = pollster::block_on(State::new(&window))?;

    for (name, actor) in actors.into_iter() {
        state.spawn_actor(name, actor, "todo".into())?;
    }

    state.add_point_light(Vec3::new(-100.0, 10.0, 10.0), random_color());
    state.add_point_light(Vec3::new(-75.0, 10.0, 0.0), random_color());
    state.add_point_light(Vec3::new(-50.0, 10.0, 10.0), random_color());
    state.add_point_light(Vec3::new(-25.0, 10.0, 0.0), random_color());
    state.add_point_light(Vec3::new(0.0, 10.0, 10.0), random_color());
    state.add_point_light(Vec3::new(25.0, 10.0, 0.0), random_color());
    state.add_point_light(Vec3::new(50.0, 10.0, 10.0), random_color());
    state.add_point_light(Vec3::new(75.0, 10.0, 0.0), random_color());
    state.add_point_light(Vec3::new(100.0, 10.0, 10.0), random_color());
    state.add_point_light(Vec3::new(125.0, 10.0, 0.0), random_color());

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
                let mut input = state.camera_state.camera_controller.get_movement_input();
                match keycode {
                    VirtualKeyCode::D => input.x = clamp(input.x + offset, -1.0, 1.0),
                    VirtualKeyCode::A => input.x = clamp(input.x - offset, -1.0, 1.0),
                    VirtualKeyCode::W => input.y = clamp(input.y + offset, -1.0, 1.0),
                    VirtualKeyCode::S => input.y = clamp(input.y - offset, -1.0, 1.0),
                    VirtualKeyCode::E => input.z = clamp(input.z + offset, -1.0, 1.0),
                    VirtualKeyCode::Q => input.z = clamp(input.z - offset, -1.0, 1.0),
                    _ => (),
                }
                state
                    .camera_state
                    .camera_controller
                    .set_movement_input(input);
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
                Err(wgpu::SurfaceError::Lost) => state.resize(state.ctx.size),
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
