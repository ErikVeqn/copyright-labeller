pub mod geo;

use std::{collections::BTreeMap, fs::File, io::Cursor, path::PathBuf, sync::Arc};

use anyhow::{anyhow, bail};
use clap::Parser;
use futures::future::{join_all, try_join_all};
use geo::Map;
use image::{DynamicImage, GenericImage, GrayImage, ImageBuffer, ImageFormat, ImageReader};
use indicatif::{ProgressBar, ProgressStyle};
use ocrs::{ImageSource, OcrEngine};
use reqwest::{Client, ClientBuilder};
use rten::Model;

const LAUNCH_YEAR: usize = 2007;
const CURRENT_YEAR: usize = 2025;

async fn download_image(
    client: &Client,
    pano_id: &str,
    zoom: u32,
    x: u32,
    y: u32,
) -> anyhow::Result<DynamicImage> {
    log::debug!("Downloading image {pano_id}@{zoom} ({x},{y})");

    let data = client.get(format!(
        "https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=apiv3&panoid={pano_id}&zoom={zoom}&x={x}&y={y}",
    )).send()
    .await?
    .bytes()
    .await?;

    Ok(ImageReader::with_format(Cursor::new(data), ImageFormat::Jpeg).decode()?)
}

type Year = usize;

type Count = usize;

#[derive(Clone, Default)]
struct Counter(BTreeMap<Year, Count>);

impl Counter {
    /// Returns the year that was found most often
    /// (year, count)
    fn best(&self) -> Option<(Year, Count)> {
        let (&year, &count) = self.0.iter().max_by_key(|&(_, v)| v)?;
        Some((year, count))
    }

    fn increment(&mut self, year: Year) {
        self.0
            .entry(year)
            .and_modify(|v| {
                *v += 1;
            })
            .or_insert(1);
    }
}

fn preprocess(image: DynamicImage) -> GrayImage {
    image.to_luma8()
}

async fn process_pano(
    pano_id: &str,
    engine: Arc<OcrEngine>,
    client: &Client,
) -> anyhow::Result<(Year, Count)> {
    let mut counter = Counter::default();

    // start at zoom level 1, because zoom level zero is "usually" bad (ocr doesn't find copyrights)
    // If we don't find any copyrights for level 1, 2 or 3, we might aswell download one more image
    // and try zoom level zero, though.
    for zoom in 1..=3 {
        let x_range = 1 << zoom;
        let y_range = ((1 << zoom) / 2).max(1);

        let tasks: Vec<_> = (0..x_range * y_range)
            .map(|i| async move {
                let x = i % x_range;
                let y = i / x_range;

                download_image(client, pano_id, zoom, x, y)
                    .await
                    .map(|image| (x, y, image))
            })
            .collect();

        let results = try_join_all(tasks).await?;
        let mut full_image = ImageBuffer::new(x_range * 512, y_range * 512);

        for (x, y, image) in results {
            full_image
                .sub_image(x * 512, y * 512, 512, 512)
                .copy_from(&image, 0, 0)
                .unwrap();
        }

        // TODO: preprocessing
        let full_image = preprocess(full_image.into());

        let engine = engine.clone();
        let lines: Vec<String> = tokio::task::spawn_blocking::<_, anyhow::Result<_>>(move || {
            let img_source = ImageSource::from_bytes(full_image.as_raw(), full_image.dimensions())?;
            let ocr_input = engine.prepare_input(img_source)?;
            let word_rects = engine.detect_words(&ocr_input)?;

            let line_rects = engine.find_text_lines(&ocr_input, &word_rects);
            let line_texts = engine.recognize_text(&ocr_input, &line_rects)?;

            Ok(line_texts
                .into_iter()
                .flatten()
                .map(|l| l.to_string())
                .collect::<Vec<_>>())
        })
        .await??;

        for line in lines {
            log::trace!("Found '{line}' on zoom level {zoom} in pano {pano_id}");

            // TODO: think about something like "actual" error correction
            for year in (LAUNCH_YEAR..=CURRENT_YEAR).filter(|year| line.contains(&year.to_string()))
            {
                log::debug!("found copyright {year} on zoom level {zoom} in pano {pano_id}");
                counter.increment(year);
            }
        }
    }

    match counter.best() {
        Some(best @ (_, best_count)) if best_count > 1 => {
            return Ok(best);
        }
        _ => {}
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

    let mut map: Map = serde_json::from_reader(File::open(&args.map)?)?;

    let detection_model = Model::load_file(PathBuf::from("models/text-detection.rten"))?;
    let recognition_model = Model::load_file(PathBuf::from("models/text-recognition.rten"))?;

    let engine = OcrEngine::new(ocrs::OcrEngineParams {
        detection_model: Some(detection_model),
        recognition_model: Some(recognition_model),
        allowed_chars: Some("1234567890 Google".to_owned()),
        ..Default::default()
    })?;
    let client = ClientBuilder::new().build()?;

    let bar = ProgressBar::new(map.locations.len() as _);
    bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})",
        )
        .unwrap()
        .progress_chars("=> "),
    );

    // tick the bar once so it shows up directly
    bar.tick();

    const BATCH_SIZE: usize = 16;

    let engine = Arc::new(engine);
    let client = Arc::new(client);
    let mut global_results = Vec::new();
    for (global_index, chunk) in map.locations.chunks(BATCH_SIZE).enumerate() {
        let mut tasks = Vec::new();

        for (local_index, location) in chunk.iter().enumerate() {
            let index = global_index * BATCH_SIZE + local_index;
            let pano_id = location
                .pano_id()
                .ok_or(anyhow!("Location {index} doesn't have pano id"))?;

            let engine = Arc::clone(&engine);
            let client = Arc::clone(&client);
            tasks.push(async move {
                process_pano(pano_id, engine, &client)
                    .await
                    .map(|counter| (index, counter))
            })
        }

        let results = join_all(tasks).await;
        for result in results.into_iter().flatten() {
            let (index, (copyright_year, count)) = result;
            global_results.push((index, copyright_year, count));
        }

        bar.inc(BATCH_SIZE as _);
    }

    bar.finish_and_clear();

    for (index, copyright, count) in global_results {
        map.locations[index]
            .extra
            .get_or_insert_default()
            .tags
            .get_or_insert_default()
            .extend_from_slice(&[format!("©{copyright}")]);
    }

    serde_json::to_writer(
        File::create(args.out.unwrap_or_else(|| PathBuf::from("out.json")))?,
        &map,
    )?;
    Ok(())
}
