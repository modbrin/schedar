use cgmath::{num_traits::clamp, *};

pub struct Camera {
    pub position: Point3<f32>,
    yaw: Rad<f32>,
    pitch: Rad<f32>,
}

impl Camera {
    pub fn new<Pos: Into<Point3<f32>>, Yaw: Into<Rad<f32>>, Pitch: Into<Rad<f32>>>(
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

    pub fn view_mat(&self) -> Matrix4<f32> {
        let (pitch_sin, pitch_cos) = self.pitch.sin_cos();
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();
        Matrix4::look_to_rh(
            self.position,
            Vector3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalize(),
            Vector3::unit_y(),
        )
    }
}

pub struct CameraController {
    rotate_delta_x: f32,
    rotate_delta_y: f32,
    rotate_speed: f32,
    raw_movement_input: Vector3<f32>,
    movement_input: Vector3<f32>,
    movement_speed: f32,
}

impl CameraController {
    pub fn new(rotate_speed: f32, movement_speed: f32) -> Self {
        Self {
            rotate_delta_x: 0.0,
            rotate_delta_y: 0.0,
            rotate_speed,
            raw_movement_input: Vector3::zero(),
            movement_input: Vector3::zero(),
            movement_speed,
        }
    }

    pub fn mouse_move(&mut self, mouse_x: f64, mouse_y: f64) {
        self.rotate_delta_x += mouse_x as f32;
        self.rotate_delta_y += mouse_y as f32;
    }

    /// Non-normalized input vector
    /// z - forward/backward, y - up/down, x - left/right
    pub fn set_movement_input(&mut self, raw_input: Vector3<f32>) {
        self.raw_movement_input = raw_input;
        if raw_input.is_zero() {
            self.movement_input = Vector3::zero();
        } else {
            self.movement_input = raw_input.normalize();
        }
    }

    pub fn get_movement_input(&mut self) -> Vector3<f32> {
        self.raw_movement_input
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: f32) {
        camera.yaw += Rad(self.rotate_delta_x) * self.rotate_speed * dt;
        camera.pitch += Rad(-self.rotate_delta_y) * self.rotate_speed * dt;

        self.rotate_delta_x = 0.0;
        self.rotate_delta_y = 0.0;

        let forward = Vector3::new(camera.yaw.cos(), 0.0, camera.yaw.sin()).normalize();
        let right = Vector3::new(-camera.yaw.sin(), 0.0, camera.yaw.cos()).normalize();

        camera.position += forward * self.movement_input.y * self.movement_speed * dt;
        camera.position += right * self.movement_input.x * self.movement_speed * dt;
        camera.position.y += self.movement_input.z * self.movement_speed * dt;

        let bound: Rad<f32> = Deg(89.0).into();
        camera.pitch = clamp(camera.pitch, -bound, bound);
    }
}
