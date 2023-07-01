use bytemuck::{Pod, Zeroable};
use encase::private::WriteInto;
use encase::{ShaderType, UniformBuffer};
use glam::*;
use std::num::NonZeroU64;

pub trait AsUniformBufferBytes {
    fn as_uniform_buffer_bytes(&self) -> Vec<u8>;
}

impl<T: ShaderType + WriteInto> AsUniformBufferBytes for T {
    fn as_uniform_buffer_bytes(&self) -> Vec<u8> {
        let mut buf = UniformBuffer::new(Vec::new());
        buf.write(&self).unwrap();
        buf.into_inner()
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

    pub fn set_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.position = Vec3::new(x, y, z);
        self
    }
    pub fn set_rotation(mut self, x: f32, y: f32, z: f32) -> Self {
        self.rotation = Vec3::new(x, y, z);
        self
    }
    pub fn set_scale(mut self, x: f32, y: f32, z: f32) -> Self {
        self.scale = Vec3::new(x, y, z);
        self
    }
}
