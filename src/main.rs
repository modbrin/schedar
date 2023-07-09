use crate::render::Actor;
use crate::primitives::Transform;
use geometry::CompositeMesh;

mod camera;
mod render;
mod error;
mod geometry;
mod primitives;
mod texture;
mod transforms;
mod utils;

fn main() {
    // "../pons-starter/assets/crytek_sponza/sponza.obj"
    // "../Sponza/sponza.obj"
    // "../pons-starter/assets/backpack/backpack.obj"
    let sponza_mesh = CompositeMesh::load_from_file("../Sponza/sponza.obj").unwrap();
    let sponza_actor = Actor::new(sponza_mesh, Transform::new().set_scale(0.1, 0.1, 0.1));

    let backpack_mesh =
        CompositeMesh::load_from_file("../pons-starter/assets/backpack/backpack.obj").unwrap();
    let backpack_actor = Actor::new(
        backpack_mesh,
        Transform::new()
            .set_position(20.0, 10.0, 0.0)
            .set_rotation(0.0, 90.0f32.to_radians(), 0.0)
            .set_scale(2.0, 2.0, 2.0),
    );

    render::run(&[("sponza", &sponza_actor), ("backpack", &backpack_actor)]).unwrap();
}
