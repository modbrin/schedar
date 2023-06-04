mod camera;
mod common;
mod error;
mod texture;
mod transforms;
mod vertex_data;

fn vertex(p: [i8; 3], n: [i8; 3], t: [i8; 2]) -> common::Vertex {
    common::Vertex {
        position: [p[0] as f32, p[1] as f32, p[2] as f32, 1.0],
        normal: [n[0] as f32, n[1] as f32, n[2] as f32, 1.0],
        uv: [t[0] as f32, t[1] as f32],
    }
}

fn create_vertices() -> Vec<common::Vertex> {
    let (pos, _col, uv, normal) = vertex_data::cube_data();
    let mut data: Vec<common::Vertex> = Vec::with_capacity(pos.len());
    for i in 0..pos.len() {
        data.push(vertex(pos[i], normal[i], uv[i]));
    }
    data.to_vec()
}

fn main() {
    let file_name = "./assets/brick.png";

    let vertex_data = create_vertices();
    let light_data = common::light([1.0, 1.0, 0.0], 0.1, 0.8, 0.4, 30.0, 1);
    let u_mode = wgpu::AddressMode::ClampToEdge;
    let v_mode = wgpu::AddressMode::ClampToEdge;

    common::run(&vertex_data, light_data, file_name, u_mode, v_mode);
}
