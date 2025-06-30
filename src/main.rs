pub mod geo;

use std::{collections::BTreeMap, fs::File, io::Cursor, path::PathBuf, sync::Arc};

use anyhow::bail;
use clap::Parser;
use futures::future::join_all;
use geo::Map;
use image::{DynamicImage, GenericImage, GrayImage, ImageBuffer, ImageFormat, ImageReader};
use indicatif::{ProgressBar, ProgressStyle};
use ocrs::{ImageSource, OcrEngine};
use reqwest::{Client, ClientBuilder};
use rten::Model;
use tokio::sync::Semaphore;

/* ───────────────────────────── constants ──────────────────────────────── */

const LAUNCH_YEAR: usize = 2007;
const CURRENT_YEAR: usize = 2025;

/// Maximum tasks (locations) processed simultaneously
const TASK_LIMIT: usize = 512;
/// Maximum concurrent HTTP tile downloads
const DOWNLOAD_LIMIT: usize = 256;
/// Maximum concurrent OCR / inference jobs
const INFERENCE_LIMIT: usize = 256;

/* ───────────────────────────── helpers ────────────────────────────────── */

async fn download_image(
    client: &Client,
    download_sem: Arc<Semaphore>,
    pano_id: &str,
    zoom: u32,
    x: u32,
    y: u32,
) -> anyhow::Result<DynamicImage> {
    let _permit = download_sem.acquire_owned().await.unwrap();

    log::debug!("Downloading image {pano_id}@{zoom} ({x},{y})");

    let data = client
        .get(format!(
            "https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=apiv3&panoid={pano_id}&zoom={zoom}&x={x}&y={y}"
        ))
        .send()
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
    fn best(&self) -> Option<(Year, Count)> {
        let (&year, &count) = self.0.iter().max_by_key(|&(_, v)| v)?;
        Some((year, count))
    }
    fn increment(&mut self, year: Year) {
        self.0.entry(year).and_modify(|v| *v += 1).or_insert(1);
    }
}

fn preprocess(img: DynamicImage) -> GrayImage {
    img.to_luma8()
}

async fn process_pano(
    pano_id: &str,
    engine: Arc<OcrEngine>,
    client: &Client,
    download_sem: Arc<Semaphore>,
    infer_sem: Arc<Semaphore>,
) -> anyhow::Result<(Year, Count)> {
    let mut counter = Counter::default();

    for &zoom in &[2] {
        let x_range = 1 << zoom;
        let y_range = ((1 << zoom) / 2).max(1);

        let mut full = ImageBuffer::new(x_range * 512, y_range * 512);

        for i in 0..x_range * y_range {
            let x = i % x_range;
            let y = i / x_range;

            let tile =
                download_image(client, Arc::clone(&download_sem), pano_id, zoom, x, y).await?;
            full = tokio::task::spawn_blocking(move || {
                full.sub_image(x * 512, y * 512, 512, 512)
                    .copy_from(&tile, 0, 0)
                    .unwrap();
                full
            })
            .await
            .unwrap();
        }

        let gray = preprocess(full.into());

        /* ---- OCR guarded by INFERENCE semaphore ------------------------- */
        let permit = infer_sem.clone().acquire_owned().await.unwrap();
        let engine_cloned = Arc::clone(&engine);

        let lines: Vec<String> = tokio::task::spawn_blocking(move || {
            let _permit = permit; // kept until this block returns

            let src = ImageSource::from_bytes(gray.as_raw(), gray.dimensions())?;
            let ocr_in = engine_cloned.prepare_input(src)?;
            let words = engine_cloned.detect_words(&ocr_in)?;

            let lines_rect = engine_cloned.find_text_lines(&ocr_in, &words);
            let texts = engine_cloned.recognize_text(&ocr_in, &lines_rect)?;

            Ok::<_, anyhow::Error>(
                texts
                    .into_iter()
                    .flatten()
                    .map(|l| l.to_string())
                    .collect::<Vec<_>>(),
            )
        })
        .await??;

        for line in lines {
            for year in (LAUNCH_YEAR..=CURRENT_YEAR).filter(|y| line.contains(&y.to_string())) {
                counter.increment(year);
            }
        }
        if let Some((_, c)) = counter.best() {
            if c > 1 {
                return Ok(counter.best().unwrap());
            }
        }
    }

    bail!("didn't find copyright")
}

/* ───────────────────────────── CLI opts ───────────────────────────────── */

#[derive(Parser)]
struct Opts {
    /// Input JSON map
    map: PathBuf,
    /// Output file (default: out.json)
    out: Option<PathBuf>,
    /// Only tag by coverage year, skip OCR
    #[arg(long, default_value = "false")]
    coverage: bool,
}

/* ───────────────────────────── main ───────────────────────────────────── */

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();
    let args = Opts::parse();

    let mut map: Map = serde_json::from_reader(File::open(&args.map)?)?;

    /* -- quick “coverage only” path ------------------------------------- */
    if args.coverage {
        map.locations.iter_mut().for_each(|loc| {
            let Some((year, _)) = loc.image_date().and_then(|d| d.split_once('-')) else {
                return;
            };

            let year = year.to_owned();

            loc.extra
                .get_or_insert_default()
                .tags
                .get_or_insert_default()
                .extend_from_slice(&[year]);
        });
        serde_json::to_writer(
            File::create(args.out.unwrap_or_else(|| PathBuf::from("out.json")))?,
            &map,
        )?;
        return Ok(());
    }

    /* -- heavy-path initialisation -------------------------------------- */
    let detection = Model::load_file("models/text-detection.rten")?;
    let recognition = Model::load_file("models/text-recognition.rten")?;

    let engine = Arc::new(OcrEngine::new(ocrs::OcrEngineParams {
        detection_model: Some(detection),
        recognition_model: Some(recognition),
        allowed_chars: Some("1234567890 Google".into()),
        ..Default::default()
    })?);
    let client = Arc::new(ClientBuilder::new().build()?);

    /* -- semaphores ------------------------------------------------------ */
    let task_sem = Arc::new(Semaphore::new(TASK_LIMIT));
    let download_sem = Arc::new(Semaphore::new(DOWNLOAD_LIMIT));
    let infer_sem = Arc::new(Semaphore::new(INFERENCE_LIMIT));

    /* -- progress bar ---------------------------------------------------- */
    let bar = ProgressBar::new(map.locations.len() as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:40.cyan/blue}] \
             {pos}/{len} • ETA {eta}",
        )
        .unwrap()
        .progress_chars("=> "),
    );
    bar.tick();

    /* -- spawn per-location tasks --------------------------------------- */
    let mut tasks = Vec::new();
    for (idx, loc) in map.locations.iter().enumerate() {
        let Some(pano_id) = loc.pano_id().map(ToString::to_string) else {
            log::warn!("location {idx} has no pano_id – skipping");
            continue;
        };

        /* limit number of simultaneous spawned tasks */
        let permit = task_sem.clone().acquire_owned().await.unwrap();

        let engine = Arc::clone(&engine);
        let client = Arc::clone(&client);
        let d_sem = Arc::clone(&download_sem);
        let i_sem = Arc::clone(&infer_sem);
        let bar = bar.clone();

        tasks.push(tokio::spawn(async move {
            /* keep permit alive for task’s lifetime */
            let _permit = permit;

            let result = process_pano(&pano_id, engine, &client, d_sem, i_sem)
                .await
                .map(|r| (idx, r));

            bar.inc(1);
            result
        }));
    }

    /* -- collect results ------------------------------------------------- */
    let mut global = Vec::<(usize, Year, Count)>::new();
    for res in join_all(tasks).await {
        if let Ok(Ok((idx, (year, cnt)))) = res {
            global.push((idx, year, cnt));
        }
    }
    bar.finish_and_clear();

    /* -- tag JSON map ---------------------------------------------------- */
    for (idx, year, _) in global {
        map.locations[idx]
            .extra
            .get_or_insert_default()
            .tags
            .get_or_insert_default()
            .extend_from_slice(&[format!("©{year}")]);
    }

    serde_json::to_writer(
        File::create(args.out.unwrap_or_else(|| PathBuf::from("out.json")))?,
        &map,
    )?;
    Ok(())
}
