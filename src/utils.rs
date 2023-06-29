use wgpu::{Device, include_spirv_raw, ShaderModule};

// returns (vertex shader, fragment shader)
pub fn load_spirv_shader(device: &Device, _name: &str) -> (ShaderModule, ShaderModule) {
    let vertex_spv = include_spirv_raw!("../shaders/out/light_vert.spv");
    let fragment_spv = include_spirv_raw!("../shaders/out/light_frag.spv");
    let shader_vert = unsafe { device.create_shader_module_spirv(&vertex_spv) };
    let shader_frag = unsafe { device.create_shader_module_spirv(&fragment_spv) };
    (shader_vert, shader_frag)
}
