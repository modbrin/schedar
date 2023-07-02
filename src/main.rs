use crate::common::Actor;
use crate::primitives::Transform;
use geometry::CompositeMesh;

mod camera;
mod common;
mod error;
mod geometry;
mod primitives;
mod texture;
mod transforms;
mod utils;

fn main() {
    // "../pons-starter/assets/crytek_sponza/sponza.obj"
    let sponza_mesh = CompositeMesh::load_from_file("../Sponza/sponza.obj").unwrap();
    let sponza_actor = Actor::new(sponza_mesh, Transform::new());

    common::run(&[("sponza".into(), &sponza_actor)]).unwrap();
}
