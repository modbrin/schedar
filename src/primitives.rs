use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vec3f {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    _padding: f32,
}

impl Vec3f {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            x,
            y,
            z,
            _padding: 0.0,
        }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn one() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vec4f {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4f {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self {
            x,
            y,
            z,
            w,
        }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    pub fn one() -> Self {
        Self::new(1.0, 1.0, 1.0, 0.0)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Color3f {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    _padding: f32,
}

impl Color3f {
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self {
            r,
            g,
            b,
            _padding: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Color4f {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

impl Color4f {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self {
            r,
            g,
            b,
            a,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Transform {
    position: Vec3f,
    rotation: Vec3f,
    scale: Vec3f,
}

impl Default for Transform {
    fn default() -> Self {
        let zero = Vec3f::new(0.0,0.0,0.0);
        let one = Vec3f::new(1.0, 1.0, 1.0);
        Self {
            position: zero,
            rotation: zero,
            scale: one,
        }
    }
}

impl Transform {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn set_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.position = Vec3f::new(x, y, z);
        self
    }
    pub fn set_rotation(mut self, x: f32, y: f32, z: f32) -> Self {
        self.rotation = Vec3f::new(x, y, z);
        self
    }
    pub fn set_scale(mut self, x: f32, y: f32, z: f32) -> Self {
        self.scale = Vec3f::new(x, y, z);
        self
    }
}