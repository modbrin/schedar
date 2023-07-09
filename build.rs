use std::process::{Command, ExitStatus};

fn main() {
    compile_to_spirv("light", "vert");
    compile_to_spirv("light", "frag");
    compile_to_spirv("light_split_base", "frag");
    compile_to_spirv("light_split_add", "frag");
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
