use std::process::Command;

fn main() {
    for (name, ty) in [
        ("light", "vert"),
        // ("light", "frag"),
        ("light_split_base", "frag"),
        ("light_split_add", "frag"),
        ("post_process", "vert"),
        ("post_process", "frag"),
        ("shadow_depth", "vert"),
        ("shadow_depth", "frag"),
        ("debug_texture", "vert"),
        ("debug_texture", "frag"),
    ] {
        compile_to_spirv(name, ty);
    }
}

fn compile_to_spirv(name: &str, ext: &str) {
    let src_path = format!("shaders/{name}.{ext}");
    let out_path = format!("shaders/out/{name}_{ext}.spv");
    let out = Command::new("glslc")
        .args([src_path.as_str(), "-o", out_path.as_str()])
        .output()
        .expect(&format!("failed to compile shader: {name}.{ext}"));
    if !out.status.success() {
        panic!("{}", String::from_utf8(out.stderr).unwrap());
    }
    println!("cargo:rerun-if-changed={}", src_path);
}
