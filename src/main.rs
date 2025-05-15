use std::{collections::HashMap, fs::File, io::Cursor, path::PathBuf};

use anyhow::anyhow;
use clap::Parser;
use image::{DynamicImage, GenericImage, GrayImage, ImageBuffer, ImageFormat, ImageReader, Rgb};
use ocrs::{ImageSource, OcrEngine};
use rten::Model;
use serde_json::Value;

#[derive(Parser)]
struct Opts {
    file: PathBuf,
}

async fn download_image(
    pano_id: &str,
    zoom: u32,
    x: u32,
    y: u32,
) -> anyhow::Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    log::debug!("Downloading image {pano_id}@{zoom} ({x},{y})");

    let data = reqwest::get(format!(
        "https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=apiv3&panoid={pano_id}&zoom={zoom}&x={x}&y={y}",
    ))
    .await?
    .bytes()
    .await?;

    let mut reader = ImageReader::new(Cursor::new(data));
    reader.set_format(ImageFormat::Jpeg);

    tokio::task::spawn_blocking(|| Ok(reader.decode()?.into_rgb8())).await?
}

fn preprocess(image_buffer: DynamicImage) -> GrayImage {
    let image_buffer = image_buffer.to_luma8();
    imageproc::filter::sharpen_gaussian(&image_buffer, 8.0, 0.8)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();
    let args = Opts::parse();

    let mut map = serde_json::from_reader::<_, Value>(File::open(&args.file)?)?;
    let locations = map
        .as_object_mut()
        .ok_or(anyhow!("invalid file format: no 'customCoordinates' field"))?["customCoordinates"]
        .as_array_mut()
        .ok_or(anyhow!(
            "invalid file format: 'customCoordinates' is not an array"
        ))?;

    let detection_model =
        Model::load_file(PathBuf::from("/home/igs/.cache/ocrs/text-detection.rten"))?;
    let recognition_model =
        Model::load_file(PathBuf::from("/home/igs/.cache/ocrs/text-recognition.rten"))?;

    let engine = OcrEngine::new(ocrs::OcrEngineParams {
        detection_model: Some(detection_model),
        recognition_model: Some(recognition_model),
        allowed_chars: Some("1234567890 Google".to_owned()),
        ..Default::default()
    })?;

    let zoom = 2;
    let x_range = 1u32 << zoom;
    let y_range = 1u32 << (zoom - 1);

    for location in locations.iter_mut() {
        let pano_id = &location["panoId"]
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

        let mut full_image = ImageBuffer::new(512 * x_range, 512 * y_range);
        let mut tasks = Vec::new();
        for y in 0..y_range {
            for x in 0..x_range {
                tasks.push(async move {
                    download_image(pano_id, zoom, x, y)
                        .await
                        .map(|image| (x, y, image))
                });
            }
        }

        let Ok(images) = futures::future::try_join_all(tasks).await else {
            log::error!("Encountered error when ");
            continue;
        };

        for (x, y, image) in images {
            full_image
                .sub_image(x * 512, y * 512, 512, 512)
                .copy_from(&image, 0, 0)?;
        }

        let img = preprocess(full_image.into());

        let img_source = ImageSource::from_bytes(img.as_raw(), img.dimensions())?;
        let ocr_input = engine.prepare_input(img_source)?;
        let word_rects = engine.detect_words(&ocr_input)?;

        let line_rects = engine.find_text_lines(&ocr_input, &word_rects);
        let line_texts = engine.recognize_text(&ocr_input, &line_rects)?;

        let mut counts = HashMap::new();
        for line in line_texts.iter().flatten() {
            for year in 2009..=2025 {
                if !line.to_string().contains(&format!("{year}")) {
                    continue;
                }

                counts
                    .entry(year)
                    .and_modify(|c| {
                        *c += 1;
                    })
                    .or_insert(1);
            }
        }

        if let Some((year, c)) = counts.iter().max_by_key(|(_, v)| **v) {
            log::info!("{pano_id}: {year} (found {c} times)");

            let tag = Value::String(format!("CR {year}"));
            location
                .as_object_mut()
                .unwrap()
                .entry("extra")
                .and_modify(|extra| {
                    extra
                        .as_object_mut()
                        .unwrap()
                        .entry("tags")
                        .and_modify(|f| {
                            f.as_array_mut().unwrap().push(tag.clone());
                        })
                        .or_insert_with(|| Value::Array(vec![tag]));
                });
        } else {
            log::info!("No copyright found for {pano_id}")
        }
    }

    serde_json::to_writer(File::create("out.json")?, &map)?;
    Ok(())
}
