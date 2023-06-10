use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use itertools::{izip, Itertools};
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;
use std::path::PathBuf;
use std::{mem, path::Path};

const TEXTURE_NOT_FOUND_COLOR: [f32; 3] = [0.99, 0.01, 0.65];

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

#[derive(Default, Debug, Clone)]
pub struct Material {
    pub albedo: Option<MaterialParam>,
    pub normal: Option<MaterialParam>,
    pub roughness: Option<MaterialParam>,
    pub metalness: Option<MaterialParam>,
    pub specular: Option<MaterialParam>,
    pub height: Option<MaterialParam>,
    pub opacity: Option<MaterialParam>,
    pub ambient_occlusion: Option<MaterialParam>,
    pub refraction: Option<MaterialParam>,
    pub emission: Option<MaterialParam>,
}

#[derive(Debug, Clone)]
pub enum MaterialParam {
    Texture(PathBuf),
    Color([f32; 3]),
    Scalar(f32),
}

impl MaterialParam {
    pub fn expect_texture(mut base_path: PathBuf, local_path: &str) -> Self {
        match Self::texture_or_none(base_path, local_path) {
            None => MaterialParam::Color(TEXTURE_NOT_FOUND_COLOR),
            Some(tex) => tex,
        }
    }
    pub fn texture_or_none(mut base_path: PathBuf, local_path: &str) -> Option<Self> {
        base_path.push(local_path);
        if fs::metadata(&base_path).is_ok() {
            Some(MaterialParam::Texture(base_path))
        } else {
            None
        }
    }
}

pub struct StaticMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

pub struct CompositeMesh {
    /// (mesh data, material_id)
    pub components: Vec<(StaticMesh, Option<usize>)>,
    /// material data, indexed by material_id
    pub materials: Vec<Material>,
}

impl CompositeMesh {
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

        let mut out_meshes = Vec::new();
        let mut out_materials = Vec::new();

        if let Ok(materials) = materials {
            let mut base_path = path.to_path_buf();
            base_path.pop();
            for mat in materials.iter() {
                let albedo = match (&mat.ambient, &mat.ambient_texture) {
                    (_, Some(local_path)) => {
                        MaterialParam::expect_texture(base_path.clone(), local_path)
                    }
                    (Some(col), None) => MaterialParam::Color(*col),
                    _ => MaterialParam::Color(TEXTURE_NOT_FOUND_COLOR),
                };
                let normal = mat.normal_texture.as_ref().and_then(|local_path| {
                    MaterialParam::texture_or_none(base_path.clone(), &local_path)
                });
                let out = Material {
                    albedo: Some(albedo),
                    normal,
                    ..Default::default()
                };
                out_materials.push(out);
            }
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
            let mat_id = model
                .mesh
                .material_id
                .filter(|id| id < &out_materials.len());
            out_meshes.push((mesh, mat_id));
        }
        let result = CompositeMesh {
            components: out_meshes,
            materials: out_materials,
        };
        Ok(result)
    }
}
