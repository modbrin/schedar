use tracing::metadata::LevelFilter;
use tracing_subscriber::prelude::*;

use crate::mesh::CompositeMesh;
use crate::primitives::Transform;
use crate::render::Actor;

mod camera;
mod error;
mod mesh;
mod primitives;
mod render;
mod texture;
mod transforms;
mod utils;

pub mod prelude {
    pub use tracing::{debug, error, info};
}

fn setup_logger(level: LevelFilter) {
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_file(true)
        .with_line_number(true)
        .with_thread_ids(true)
        .with_filter(level);
    tracing_subscriber::registry().with(fmt_layer).init();
}

fn main() {
    setup_logger(LevelFilter::WARN);

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
