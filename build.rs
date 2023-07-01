use std::process::{Command, ExitStatus};

static TO_COMPILE: &[&str] = &["light"];

fn main() {
    for name in TO_COMPILE {
        compile_to_spirv(name);
    }
}

fn compile_to_spirv(name: &str) {
    let vert_src = format!("shaders/{}.vert", name);
    let frag_src = format!("shaders/{}.frag", name);
    let vert_out = format!("shaders/out/{}_vert.spv", name);
    let frag_out = format!("shaders/out/{}_frag.spv", name);
    // glslc shaders/light.vert -o shaders/out/light_vert.spv
    // glslc shaders/light.frag -o shaders/out/light_frag.spv
    let out = Command::new("glslc")
        .args([vert_src.as_str(), "-o", vert_out.as_str()])
        .output()
        .expect(&format!("failed to compile vertex shader: {}", name));
    if !out.status.success() {
        panic!("{}", String::from_utf8(out.stderr).unwrap());
    }
    let out = Command::new("glslc")
        .args([frag_src.as_str(), "-o", frag_out.as_str()])
        .output()
        .expect(&format!("failed to compile fragment shader: {}", name));
    if !out.status.success() {
        panic!("{}", String::from_utf8(out.stderr).unwrap());
    }
    println!("cargo:rerun-if-changed={}", vert_src);
    println!("cargo:rerun-if-changed={}", frag_src);
}
