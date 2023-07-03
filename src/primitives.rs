use bytemuck::{Pod, Zeroable};
use encase::private::WriteInto;
use encase::{ShaderType, UniformBuffer};
use glam::*;
use std::num::NonZeroU64;

pub trait ShaderTypeExt {
    fn as_uniform_buffer_bytes(&self) -> Vec<u8>;
}

impl<T: ShaderType + WriteInto> ShaderTypeExt for T {
    fn as_uniform_buffer_bytes(&self) -> Vec<u8> {
        let mut buf = UniformBuffer::new(Vec::new());
        buf.write(&self).unwrap();
        buf.into_inner()
    }
}

pub trait ShaderTypeDefaultExt {
    fn default_size() -> u64;
}

impl<T: ShaderType + WriteInto + Default> ShaderTypeDefaultExt for T {
    fn default_size() -> u64 {
        T::default().size().get()
    }
}

#[derive(Clone, Debug)]
pub struct Transform {
    position: Vec3,
    rotation: Vec3,
    scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::zeroed(),
            rotation: Vec3::zeroed(),
            scale: Vec3::new(1.0, 1.0, 1.0),
        }
    }
}

impl Transform {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn get_position(&self) -> Vec3 {
        self.position
    }

    pub fn get_rotation(&self) -> Vec3 {
        self.rotation
    }

    pub fn get_scale(&self) -> Vec3 {
        self.scale
    }

    pub fn set_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.position = Vec3::new(x, y, z);
        self
    }

    /// Set euler rotation in radians
    pub fn set_rotation(mut self, x: f32, y: f32, z: f32) -> Self {
        self.rotation = Vec3::new(x, y, z);
        self
    }

    pub fn set_scale(mut self, x: f32, y: f32, z: f32) -> Self {
        self.scale = Vec3::new(x, y, z);
        self
    }
}
