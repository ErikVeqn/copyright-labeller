use std::collections::HashMap;
use std::{fs::File, io::Cursor, path::PathBuf, sync::Arc};

use anyhow::{anyhow, bail};
use clap::Parser;
use futures::future::{join_all, try_join_all};
use image::{DynamicImage, GenericImage, GrayImage, ImageBuffer, ImageFormat, ImageReader};
use indicatif::{ProgressBar, ProgressStyle};
use ocrs::{ImageSource, OcrEngine};
use rten::Model;
use serde_json::Value;

const LAUNCH_YEAR: usize = 2007;
const CURRENT_YEAR: usize = 2025;

async fn download_image(pano_id: &str, zoom: u32, x: u32, y: u32) -> anyhow::Result<DynamicImage> {
    log::trace!("Downloading image {pano_id}@{zoom} ({x},{y})");

    let data = reqwest::get(format!(
        "https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=apiv3&panoid={pano_id}&zoom={zoom}&x={x}&y={y}",
    ))
    .await?
    .bytes()
    .await?;

    tokio::task::spawn_blocking(|| {
        Ok(ImageReader::with_format(Cursor::new(data), ImageFormat::Jpeg).decode()?)
    })
    .await?
}

fn preprocess(image_buffer: DynamicImage) -> GrayImage {
    let image_buffer = image_buffer.to_luma8();
    imageproc::filter::sharpen_gaussian(&image_buffer, 8.0, 0.8)
}

#[derive(Copy, Clone, Default)]
struct Counter([u8; 32]);

impl Counter {
    fn year(&self) -> usize {
        let Some(max_idx) = (0..32).max_by_key(|&i| self.0[i]) else {
            unreachable!("iterator is never empty and thus always has maximum");
        };

        max_idx + LAUNCH_YEAR
    }

    fn max(&self) -> u8 {
        let Some(&max) = self.0.iter().max() else {
            unreachable!("iterator is never empty and thus always has maximum");
        };

        max
    }

    fn increment(&mut self, year: usize) {
        self.0[year - LAUNCH_YEAR] += 1;
    }
}

async fn process_pano(pano_id: &str, engine: Arc<OcrEngine>) -> anyhow::Result<Counter> {
    let mut counter = Counter::default();

    for zoom in [1, 2, 3, 0] {
        let x_range = 1 << zoom;
        let y_range = ((1 << zoom) / 2).max(1);

        let tasks: Vec<_> = (0..x_range * y_range)
            .map(|i| async move {
                let x = (i % x_range) as u32;
                let y = (i / x_range) as u32;

                download_image(pano_id, zoom, x, y)
                    .await
                    .map(|image| (x, y, image))
            })
            .collect();

        let results = try_join_all(tasks).await?;
        let mut full_image = ImageBuffer::new(x_range * 512, y_range * 512);
        for (x, y, image) in results {
            full_image
                .sub_image(x * 512, y * 512, 512, 512)
                .copy_from(&image, 0, 0)?;
        }

        let engine = engine.clone();
        let lines = {
            let img_source = ImageSource::from_bytes(full_image.as_raw(), full_image.dimensions())?;
            let ocr_input = engine.prepare_input(img_source)?;
            let word_rects = engine.detect_words(&ocr_input)?;

            let line_rects = engine.find_text_lines(&ocr_input, &word_rects);
            let line_texts = engine.recognize_text(&ocr_input, &line_rects)?;

            line_texts.into_iter().flatten().collect::<Vec<_>>()
        };

        // TODO: think about something like "actual" error correction
        for line in lines {
            let line = line.to_string();

            log::debug!("Found '{line}' on zoom level {zoom} in pano {pano_id}");

            for year in LAUNCH_YEAR..=CURRENT_YEAR {
                if line.contains(&format!("{year}")) {
                    log::info!("found copyright {year} on zoom level {zoom} in pano {pano_id}");
                    counter.increment(year);
                }
            }
        }

        if counter.max() > 1 {
            return Ok(counter);
        }
    }

    bail!("didn't find copyright")
}

#[derive(Parser)]
struct Opts {
    /// the json to add copyright labels to
    map: PathBuf,
    /// the location to store the labelled map in. If no location is specified, './out.json' is used
    out: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();
    let args = Opts::parse();

    let mut map = serde_json::from_reader::<_, Value>(File::open(&args.map)?)?;
    let locations = map
        .as_object()
        .ok_or(anyhow!("invalid file format: no 'customCoordinates' field"))?["customCoordinates"]
        .as_array()
        .ok_or(anyhow!(
            "invalid file format: 'customCoordinates' is not an array"
        ))?;

    let detection_model = Model::load_file(PathBuf::from("models/text-detection.rten"))?;
    let recognition_model = Model::load_file(PathBuf::from("models/text-recognition.rten"))?;

    let engine = OcrEngine::new(ocrs::OcrEngineParams {
        detection_model: Some(detection_model),
        recognition_model: Some(recognition_model),
        allowed_chars: Some("1234567890 Google".to_owned()),
        ..Default::default()
    })?;

    let bar = ProgressBar::new(locations.len() as _);
    let sty = ProgressStyle::with_template("{bar:40.green/yellow} {pos:>7}/{len:7}").unwrap();
    bar.set_style(sty);

    let mut global_results = Vec::new();
    let engine = Arc::new(engine);
    const CHUNK_SIZE: usize = 4;
    for (global_index, chunk) in locations.chunks(CHUNK_SIZE).enumerate() {
        let mut tasks = Vec::new();

        for (local_index, location) in chunk.into_iter().enumerate() {
            let pano_id = location["panoId"]
                .as_str()
                .ok_or(anyhow!("location doesn't have a pano id"))
                .or_else::<anyhow::Error, _>(|_| {
                    Ok(location
                        .get("extra")
                        .ok_or(anyhow!("location doesn't have 'panoId' tag _nor_ 'extra'"))?
                        .get("panoId")
                        .ok_or(anyhow!("location doesn't have 'panoId' nor 'extra/panoId'"))?
                        .as_str()
                        .unwrap())
                })?;

            let engine = Arc::clone(&engine);
            tasks.push(async move {
                process_pano(&pano_id, engine)
                    .await
                    .map(|counter| (global_index * CHUNK_SIZE + local_index, counter))
            })
        }

        let results = join_all(tasks).await;
        for result in results {
            if let Ok((index, counter)) = result {
                global_results.push((index, counter.year(), counter.max()));
            }
        }

        bar.inc(CHUNK_SIZE as _);
    }

    bar.finish();

    for (index, copyright, count) in global_results {
        let custom_coordinates = map["customCoordinates"].as_array_mut().unwrap();
        let loc = custom_coordinates[index].as_object_mut().unwrap();
        if !loc.contains_key("extra") {
            loc.insert("extra".into(), Value::Object(Default::default()));
        }

        let extra = loc["extra"].as_object_mut().unwrap();

        extra
            .entry("tags")
            .and_modify(|tags| {
                tags.as_array_mut().unwrap().append(&mut vec![
                    Value::String(format!("CR {copyright}")),
                    Value::String(format!("CR Count {count}")),
                ]);
            })
            .or_insert(Value::Array(vec![
                Value::String(format!("CR {copyright}")),
                Value::String(format!("CR Count {count}")),
            ]));
    }

    serde_json::to_writer(
        File::create(args.out.unwrap_or_else(|| PathBuf::from("out.json")))?,
        &map,
    )?;
    Ok(())
}
