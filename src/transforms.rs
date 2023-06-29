#![allow(dead_code)]
use cgmath::*;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

pub fn create_projection(aspect: f32, is_perspective: bool) -> Matrix4<f32> {
    let project_mat: Matrix4<f32>;
    if is_perspective {
        project_mat = OPENGL_TO_WGPU_MATRIX * perspective(Deg(60.0), aspect, 0.1, 500.0);
    } else {
        project_mat = OPENGL_TO_WGPU_MATRIX * ortho(-4.0, 4.0, -3.0, 3.0, -1.0, 6.0);
    }
    project_mat
}

pub fn create_view_projection(
    camera_position: Point3<f32>,
    look_direction: Point3<f32>,
    up_direction: Vector3<f32>,
    aspect: f32,
    is_perspective: bool,
) -> (Matrix4<f32>, Matrix4<f32>, Matrix4<f32>) {
    let view_mat = Matrix4::look_at_rh(camera_position, look_direction, up_direction);
    let project_mat = create_projection(aspect, is_perspective);
    let view_project_mat = project_mat * view_mat;
    (view_mat, project_mat, view_project_mat)
}

pub fn create_transforms(
    translation: Point3<f32>,
    rotation: Point3<f32>,
    scaling: Point3<f32>,
) -> Matrix4<f32> {
    let trans_mat =
        Matrix4::from_translation(Vector3::new(translation.x, translation.y, translation.z));
    let rotate_mat_x = Matrix4::from_angle_x(Rad(rotation.x));
    let rotate_mat_y = Matrix4::from_angle_y(Rad(rotation.y));
    let rotate_mat_z = Matrix4::from_angle_z(Rad(rotation.z));
    let scale_mat = Matrix4::from_nonuniform_scale(scaling.x, scaling.y, scaling.z);
    let model_mat = trans_mat * rotate_mat_z * rotate_mat_y * rotate_mat_x * scale_mat;
    model_mat
}