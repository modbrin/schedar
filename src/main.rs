use geometry::CompositeMesh;
use crate::common::Actor;
use crate::primitives::Transform;

mod camera;
mod common;
mod error;
mod geometry;
mod texture;
mod transforms;
mod utils;
mod primitives;

fn main() {
    let sponza_mesh = CompositeMesh::load_from_file("../Sponza/sponza.obj").unwrap();
    let sponza_actor = Actor::new(sponza_mesh, Transform::new());

    common::run(&[("sponza".into(), &sponza_actor)]).unwrap();
}
