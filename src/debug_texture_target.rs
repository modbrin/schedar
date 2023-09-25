use anyhow::Result;

use crate::utils;

pub struct DebugTextureTarget {
    pub render_pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub sampler: wgpu::Sampler,
    pub texture_format: wgpu::TextureFormat,
    pub shader_vert: wgpu::ShaderModule,
    pub shader_frag: wgpu::ShaderModule,
}

impl DebugTextureTarget {
    pub fn new(
        device: &wgpu::Device,
        texture_format: wgpu::TextureFormat,
        debug_depth_texture_view: &wgpu::TextureView,
    ) -> Result<Self> {
        let shader_vert = utils::load_spirv_shader_module(
            &device,
            "base_vert",
            "./shaders/out/debug_texture_vert.spv",
        )?;
        let shader_frag = utils::load_spirv_shader_module(
            &device,
            "base_frag",
            "./shaders/out/debug_texture_frag.spv",
        )?;
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("schedar.sampler.debug_texture"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let bind_group_layout = Self::init_bind_group_layout(device);
        let bind_group = Self::init_bind_group(
            device,
            &bind_group_layout,
            &debug_depth_texture_view,
            &sampler,
        );
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
            sampler,
            texture_format,
            shader_vert,
            shader_frag,
        };
        Ok(out)
    }

    pub fn init_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                utils::make_depth_bgl_entry_texture(0),
                utils::make_common_bgl_entry_sampler(1),
            ],
            label: Some("schedar.bind_group_layout.debug_texture"),
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
            label: Some("schedar.bind_group.debug_texture"),
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
            label: Some("schedar.render_pipeline_layout.debug_texture"),
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
            label: Some("schedar.render_pipeline.debug_texture"),
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

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        out_tex: &wgpu::Texture,
        depth_bind_group: &wgpu::BindGroup,
    ) {
        {
            let out_tex_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("schedar.render_pass.debug_texture"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &out_tex_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &depth_bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }
    }
}
