use anyhow::Result;
use encase::ShaderType;
use glam::Mat4;

use crate::primitives::ShaderTypeDefaultExt;
use crate::utils;

#[derive(Clone, Copy, Default, ShaderType)]
pub struct ShadowPushConst {
    pub light_space_mat: Mat4,
    pub model_mat: Mat4,
}

pub struct ShadowTarget {
    pub render_pipeline: wgpu::RenderPipeline,
    pub depth_texture: wgpu::Texture,
    pub depth_texture_view: wgpu::TextureView,
    pub shader_vert: wgpu::ShaderModule,
    pub shader_frag: wgpu::ShaderModule,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub sampler: wgpu::Sampler,
    pub width: u32,
    pub height: u32,
}

impl ShadowTarget {
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        let shader_vert = utils::load_spirv_shader_module(
            &device,
            "base_vert",
            "./shaders/out/shadow_depth_vert.spv",
        )?;
        let shader_frag = utils::load_spirv_shader_module(
            &device,
            "base_frag",
            "./shaders/out/shadow_depth_frag.spv",
        )?;
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("schedar.sampler.shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let (width, height) = (2048, 2048);
        let (depth_texture, depth_texture_view) = Self::init_depth_texture(device, width, height);
        let bind_group_layout = Self::init_bind_group_layout(device);
        let bind_group =
            Self::init_bind_group(device, &bind_group_layout, &depth_texture_view, &sampler);
        let render_pipeline = Self::init_render_pipeline(device, &shader_vert, &shader_frag);
        let result = Self {
            render_pipeline,
            depth_texture,
            depth_texture_view,
            shader_vert,
            shader_frag,
            bind_group_layout,
            bind_group,
            sampler,
            width,
            height,
        };
        Ok(result)
    }

    pub fn init_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus, // TODO: Depth32Float
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            label: Some("schedar.texture.shadow.depth"),
            view_formats: &[],
        });
        let texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("schedar.texture_view.shadow.depth"),
            ..Default::default()
        });
        (depth_texture, texture_view)
    }

    pub fn init_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                // utils::make_common_bgl_entry_uniform(0),
                utils::make_depth_bgl_entry_texture(0),
                utils::make_common_bgl_entry_sampler(1),
            ],
            label: Some("schedar.shadow_target.bind_group_layout"),
        })
    }
    pub fn init_bind_group(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        depth_target: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        let bind_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&depth_target),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ];
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &bind_entries,
            label: Some("schedar.shadow_target.bind_group"),
        });
        texture_bind_group
    }

    pub fn init_render_pipeline(
        device: &wgpu::Device,
        shader_vert: &wgpu::ShaderModule,
        shader_frag: &wgpu::ShaderModule,
    ) -> wgpu::RenderPipeline {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("schedar.render_pipeline_layout.shadow"),
            bind_group_layouts: &[],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::VERTEX,
                range: 0..ShadowPushConst::default_size() as u32,
            }],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("schedar.shadow_target.render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_vert,
                entry_point: "main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_frag,
                entry_point: "main",
                targets: &[],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus, // TODO: Depth32Float
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        pipeline
    }
}
