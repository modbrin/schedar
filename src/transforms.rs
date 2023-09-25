use glam::*;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Mat4 = Mat4::from_cols_array(
    &[
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.0,
        0.0, 0.0, 0.5, 1.0,
    ]
);

pub fn create_projection(aspect: f32, is_perspective: bool) -> Mat4 {
    let project_mat: Mat4;
    if is_perspective {
        project_mat =
            OPENGL_TO_WGPU_MATRIX * Mat4::perspective_rh(60.0f32.to_radians(), aspect, 0.1, 600.0);
    } else {
        project_mat =
            OPENGL_TO_WGPU_MATRIX * Mat4::orthographic_rh(-4.0, 4.0, -3.0, 3.0, -1.0, 100.0);
    }
    project_mat
}

pub fn create_view_projection(
    camera_position: Vec3,
    look_direction: Vec3,
    up_direction: Vec3,
    aspect: f32,
    is_perspective: bool,
) -> (Mat4, Mat4, Mat4) {
    let view_mat = Mat4::look_at_rh(camera_position, look_direction, up_direction);
    let project_mat = create_projection(aspect, is_perspective);
    let view_project_mat = project_mat * view_mat;
    (view_mat, project_mat, view_project_mat)
}

pub fn create_transforms(translation: Vec3, rotation: Vec3, scaling: Vec3) -> Mat4 {
    let trans_mat = Mat4::from_translation(translation);
    let rotate_mat_x = Mat4::from_rotation_x(rotation.x);
    let rotate_mat_y = Mat4::from_rotation_y(rotation.y);
    let rotate_mat_z = Mat4::from_rotation_z(rotation.z);
    let scale_mat = Mat4::from_scale(scaling);
    let model_mat = trans_mat * rotate_mat_z * rotate_mat_y * rotate_mat_x * scale_mat;
    model_mat
}
