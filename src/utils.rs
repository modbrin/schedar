use num::Num;
use std::ops::{BitAnd, Not};
use wgpu::{include_spirv_raw, Device, ShaderModule};

// returns (vertex shader, fragment shader)
pub fn load_spirv_shader(device: &Device, _name: &str) -> (ShaderModule, ShaderModule) {
    let vertex_spv = include_spirv_raw!("../shaders/out/light_vert.spv");
    let fragment_spv = include_spirv_raw!("../shaders/out/light_frag.spv");
    let shader_vert = unsafe { device.create_shader_module_spirv(&vertex_spv) };
    let shader_frag = unsafe { device.create_shader_module_spirv(&fragment_spv) };
    (shader_vert, shader_frag)
}

pub fn round_to_next_multiple<T: Num + Copy>(number: T, multiple: T) -> T {
    (((number + multiple - T::one()) / multiple) * multiple)
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
