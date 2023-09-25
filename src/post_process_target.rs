use anyhow::Result;

use crate::utils;

pub struct PostProcessTarget {
    pub render_pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub color_texture: wgpu::Texture,
    pub color_texture_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub texture_format: wgpu::TextureFormat,
    pub shader_vert: wgpu::ShaderModule,
    pub shader_frag: wgpu::ShaderModule,
    pub width: u32,
    pub height: u32,
}

impl PostProcessTarget {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        texture_format: wgpu::TextureFormat,
    ) -> Result<Self> {
        let shader_vert = utils::load_spirv_shader_module(
            &device,
            "base_vert",
            "./shaders/out/post_process_vert.spv",
        )?;
        let shader_frag = utils::load_spirv_shader_module(
            &device,
            "base_frag",
            "./shaders/out/post_process_frag.spv",
        )?;
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("schedar.sampler.post_process"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let (texture, color_target) =
            Self::init_color_target(device, width, height, texture_format);
        let bind_group_layout = Self::init_bind_group_layout(device);
        let bind_group = Self::init_bind_group(device, &bind_group_layout, &color_target, &sampler);
        let render_pipeline = Self::init_render_pipeline(
            device,
            &bind_group_layout,
            &shader_vert,
            &shader_frag,
            texture_format,
        );
        let out = Self {
            render_pipeline,
            bind_group_layout,
            bind_group,
            color_texture: texture,
            color_texture_view: color_target,
            sampler,
            texture_format,
            shader_vert,
            shader_frag,
            width,
            height,
        };
        Ok(out)
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        let (texture, texture_view) =
            Self::init_color_target(device, width, height, self.texture_format);
        self.color_texture = texture;
        self.color_texture_view = texture_view;
        self.bind_group = Self::init_bind_group(
            device,
            &self.bind_group_layout,
            &self.color_texture_view,
            &self.sampler,
        );
    }

    pub fn init_color_target(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture_desc = wgpu::TextureDescriptor {
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            label: None,
            view_formats: &[],
        };
        let texture = device.create_texture(&texture_desc);
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("schedar.color_target.view"),
            ..Default::default()
        });
        (texture, texture_view)
    }
    pub fn init_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                // utils::make_common_bgl_entry_uniform(0),
                utils::make_common_bgl_entry_texture(0),
                utils::make_common_bgl_entry_sampler(1),
            ],
            label: Some("schedar.bind_group_layout.post_process"),
        })
    }
    pub fn init_bind_group(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        color_target: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        let bind_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&color_target),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ];
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &bind_entries,
            label: Some("schedar.bind_group.post_process"),
        });
        texture_bind_group
    }

    pub fn init_render_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        shader_vert: &wgpu::ShaderModule,
        shader_frag: &wgpu::ShaderModule,
        texture_format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("schedar.render_pipeline_layout.post_process"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let color_target_state = wgpu::ColorTargetState {
            format: texture_format,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent::REPLACE,
                alpha: wgpu::BlendComponent::REPLACE,
            }),
            write_mask: wgpu::ColorWrites::ALL,
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("schedar.render_pipeline.post_process"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_vert,
                entry_point: "main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_frag,
                entry_point: "main",
                targets: &[Some(color_target_state)],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        pipeline
    }

    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, out_tex: &wgpu::Texture) {
        {
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &out_tex,
                    mip_level: 0,
                    aspect: wgpu::TextureAspect::All,
                    origin: wgpu::Origin3d::ZERO,
                },
                wgpu::ImageCopyTexture {
                    texture: &self.color_texture,
                    mip_level: 0,
                    aspect: wgpu::TextureAspect::All,
                    origin: wgpu::Origin3d::ZERO,
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
        }
        {
            let out_tex_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default()); //&self.color_target,
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("schedar.render_pass.post_process"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &out_tex_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }
    }
}
