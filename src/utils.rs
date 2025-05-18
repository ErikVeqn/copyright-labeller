use std::f32::consts::PI;

use image::{DynamicImage, GenericImageView, Pixel, Rgb, RgbImage};
use nalgebra::{Vector2, Vector3};

fn rotate_xy(p: Vector3<f32>, angle: Vector2<f32>) -> Vector3<f32> {
    let c = Vector2::new(angle.x.cos(), angle.y.cos());
    let s = Vector2::new(angle.x.sin(), angle.y.sin());

    let p = Vector3::new(p.x, c.x * p.y + s.x * p.z, -s.x * p.y + c.x * p.z);
    Vector3::new(c.y * p.x + s.y * p.z, p.y, -s.y * p.x + c.y * p.z)
}

pub fn equirectangular_to_rectilinear(
    pano: &DynamicImage,
    fov: f32,
    yaw: f32,
    pitch: f32,
    out_width: u32,
    out_height: u32,
) -> RgbImage {
    let (pano_width, pano_height) = pano.dimensions();

    let aspect_ratio = out_width as f32 / out_height as f32;
    let fov_h = fov;
    let fov_v = fov / aspect_ratio;

    RgbImage::from_fn(out_width, out_height, |x, y| {
        let uv = Vector2::new(
            2.0 * (x as f32 + 0.5) / out_width as f32 - 1.0,
            2.0 * (y as f32 + 0.5) / out_height as f32 - 1.0,
        );

        let tan_half_fov = Vector2::new((0.5 * fov_h).tan(), (0.5 * fov_v).tan());
        let mut dir = Vector3::new(uv.x * tan_half_fov.x, uv.y * tan_half_fov.y, 1.0).normalize();

        let (cp, sp) = (pitch.cos(), pitch.sin());
        let (cy, sy) = (yaw.cos(), yaw.sin());

        // Rotate pitch (X axis)
        let dy = cp * dir.y - sp * dir.z;
        let dz = sp * dir.y + cp * dir.z;
        dir.y = dy;
        dir.z = dz;

        // Rotate yaw (Y axis)
        let dx = cy * dir.x + sy * dir.z;
        let dz = -sy * dir.x + cy * dir.z;
        dir.x = dx;
        dir.z = dz;

        // Spherical coordinates
        let theta = dir.z.atan2(dir.x); // [-π, π]
        let phi = (-dir.y).acos(); // [0, π]

        // Normalized texture coordinates
        let u = (theta + PI) / (2.0 * PI);
        let v = phi / PI;

        let px = (u * pano_width as f32).rem_euclid(pano_width as f32);
        let py = (v * pano_height as f32).clamp(0.0, pano_height as f32 - 1.0);

        let color = pano.get_pixel(px as u32, py as u32).to_rgb();

        Rgb([color[0], color[1], color[2]])
    })
}
