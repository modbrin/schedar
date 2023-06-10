use geometry::StaticMesh;

mod camera;
mod common;
mod error;
mod geometry;
mod texture;
mod transforms;

fn main() {
    let file_name = "./assets/brick.png";

    let monkey = StaticMesh::load_from_file("assets/monkey.obj").unwrap();

    let light_data = common::light([1.0, 1.0, 0.0], 0.1, 0.8, 0.4, 30.0);
    let u_mode = wgpu::AddressMode::ClampToEdge;
    let v_mode = wgpu::AddressMode::ClampToEdge;

    common::run(&monkey, light_data, file_name, u_mode, v_mode).unwrap();
}
