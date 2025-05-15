pub enum Location {
    LatLng { lat: f32, lng: f32 },
    PanoID { pano_id: String },
}
