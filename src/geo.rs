use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, Copy, Clone, Debug)]
pub struct Coordinates {
    pub lat: f32,
    pub lng: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct Extras {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pano_id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub pano_date: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Location {
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub coordinates: Option<Coordinates>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub heading: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub pitch: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub zoom: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<Extras>,

    // for these fields it doesn't matter if they serialize to "null". map-making.app does this
    // aswell, so we want to support this
    pub pano_id: Option<String>,
    pub country_code: Option<String>,
    pub state_code: Option<String>,
}

impl Location {
    /// convenience method for checking for pano_id both in [`Location::pano_id`]: and
    /// [`Location::extra`]
    pub fn pano_id(&self) -> Option<&str> {
        let extra_pano_id = self
            .extra
            .as_ref()
            .and_then(|extra| extra.pano_id.as_deref());
        self.pano_id.as_deref().or(extra_pano_id)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Map {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(rename = "customCoordinates")]
    pub locations: Vec<Location>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}
