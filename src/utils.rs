use anyhow::*;
use num::{clamp, Num};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read};
use std::ops::{BitAnd, Not};
use std::path::Path;
use wgpu::{include_spirv_raw, Device, ShaderModule, ShaderModuleDescriptorSpirV};

// // returns (vertex shader, fragment shader)
// pub fn load_spirv_shader(device: &Device) -> (ShaderModule, ShaderModule) {
//     let vertex_spv = include_spirv_raw!("../shaders/out/light_vert.spv");
//     let fragment_spv = include_spirv_raw!("../shaders/out/light_frag.spv");
//     let shader_vert = unsafe { device.create_shader_module_spirv(&vertex_spv) };
//     let shader_frag = unsafe { device.create_shader_module_spirv(&fragment_spv) };
//     (shader_vert, shader_frag)
// }

pub fn load_spirv_shader_module(
    device: &Device,
    name: &str,
    path: impl AsRef<Path>,
) -> Result<ShaderModule> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut data = Vec::new();
    reader.read_to_end(&mut data)?;
    let descriptor = ShaderModuleDescriptorSpirV {
        label: Some(name),
        source: wgpu::util::make_spirv_raw(&data),
    };
    let module = unsafe { device.create_shader_module_spirv(&descriptor) };
    Ok(module)
}

pub fn round_to_next_multiple<T: Num + Copy>(number: T, multiple: T) -> T {
    ((number + multiple - T::one()) / multiple) * multiple
}

/// Maps color scalar from 0.0..1.0 to 0..255 range
pub fn color_f32_to_u8(val: f32) -> u8 {
    (256.0 * clamp(val, 0.0, 1.0)) as u8
}

#[cfg(test)]
mod tests {
    use crate::utils::round_to_next_multiple;

    #[test]
    fn test_round_to_multiple() {
        assert_eq!(round_to_next_multiple(45, 16), 48);
        assert_eq!(round_to_next_multiple(0, 3), 0);
        assert_eq!(round_to_next_multiple(1, 3), 3);
        assert_eq!(round_to_next_multiple(2, 3), 3);
        assert_eq!(round_to_next_multiple(3, 3), 3);
        assert_eq!(round_to_next_multiple(4, 3), 6);
        assert_eq!(round_to_next_multiple(3000001, 3), 3000003);
    }
}
