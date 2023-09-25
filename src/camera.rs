use bytemuck::Zeroable;
use glam::*;
use num::clamp;

pub const UP_DIRECTION: Vec3 = Vec3::new(0.0, 1.0, 0.0);

pub struct Camera {
    pub position: Vec3,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    pub fn new<Pos: Into<Vec3>, Yaw: Into<f32>, Pitch: Into<f32>>(
        position: Pos,
        yaw: Yaw,
        pitch: Pitch,
    ) -> Self {
        Self {
            position: position.into(),
            yaw: yaw.into(),
            pitch: pitch.into(),
        }
    }

    pub fn view_mat(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.direction(), UP_DIRECTION)
    }

    pub fn direction(&self) -> Vec3 {
        let (pitch_sin, pitch_cos) = self.pitch.sin_cos();
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();
        Vec3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalize()
    }
}

pub struct CameraController {
    rotate_delta_x: f32,
    rotate_delta_y: f32,
    rotate_speed: f32,
    raw_movement_input: Vec3,
    movement_input: Vec3,
    movement_speed: f32,
}

impl CameraController {
    pub fn new(rotate_speed: f32, movement_speed: f32) -> Self {
        Self {
            rotate_delta_x: 0.0,
            rotate_delta_y: 0.0,
            rotate_speed,
            raw_movement_input: Vec3::zeroed(),
            movement_input: Vec3::zeroed(),
            movement_speed,
        }
    }

    pub fn mouse_move(&mut self, mouse_x: f64, mouse_y: f64) {
        self.rotate_delta_x += mouse_x as f32;
        self.rotate_delta_y += mouse_y as f32;
    }

    /// Non-normalized input vector
    /// z - forward/backward, y - up/down, x - left/right
    pub fn set_movement_input(&mut self, raw_input: Vec3) {
        self.raw_movement_input = raw_input;
        self.movement_input = raw_input.normalize_or_zero();
    }

    pub fn get_movement_input(&mut self) -> Vec3 {
        self.raw_movement_input
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: f32) {
        camera.yaw += self.rotate_delta_x * self.rotate_speed * dt;
        camera.pitch += -self.rotate_delta_y * self.rotate_speed * dt;

        self.rotate_delta_x = 0.0;
        self.rotate_delta_y = 0.0;

        let forward = Vec3::new(camera.yaw.cos(), 0.0, camera.yaw.sin()).normalize();
        let right = Vec3::new(-camera.yaw.sin(), 0.0, camera.yaw.cos()).normalize();

        camera.position += forward * self.movement_input.y * self.movement_speed * dt;
        camera.position += right * self.movement_input.x * self.movement_speed * dt;
        camera.position.y += self.movement_input.z * self.movement_speed * dt;

        let bound: f32 = 89.0f32.to_radians();
        camera.pitch = clamp(camera.pitch, -bound, bound);
    }
}
