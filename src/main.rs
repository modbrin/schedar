use geometry::CompositeMesh;

mod camera;
mod common;
mod error;
mod geometry;
mod texture;
mod transforms;

fn main() {
    let monkey = CompositeMesh::load_from_file("../Sponza/sponza.obj").unwrap();
    let light_data = common::light([1.0, 1.0, 1.0], 0.1, 0.8, 0.4, 30.0);

    common::run(&monkey, light_data).unwrap();
}
