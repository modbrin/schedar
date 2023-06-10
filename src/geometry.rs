use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use itertools::{izip, Itertools};
use std::fmt::Debug;
use std::{mem, path::Path};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0=>Float32x3, 1=>Float32x3, 2=>Float32x2];
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

pub struct StaticMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl StaticMesh {
    pub fn load_from_file<P>(path: P) -> Result<Self>
    where
        P: AsRef<Path> + Debug,
    {
        let path = path.as_ref();
        let (models, materials) = tobj::load_obj(
            path,
            &tobj::LoadOptions {
                single_index: true,
                triangulate: true,
                ignore_lines: true,
                ignore_points: true,
                ..Default::default()
            },
        )?;
        let materials = materials?;
        if models.len() > 1 {
            return Err(anyhow!(
                "muliple meshes per obj file is not supported yet: {path:?}"
            ));
        }
        for model in models.iter() {
            let mut vertices = Vec::new();
            for (pos, norm, uv) in izip!(
                model.mesh.positions.chunks_exact(3),
                model.mesh.normals.chunks_exact(3),
                model.mesh.texcoords.chunks_exact(2)
            ) {
                vertices.push(Vertex {
                    position: pos.try_into()?,
                    normal: norm.try_into()?,
                    uv: uv.try_into()?,
                });
            }
            let mesh = StaticMesh {
                vertices,
                indices: model.mesh.indices.clone(),
            };
            return Ok(mesh);
        }
        todo!()
    }
}
