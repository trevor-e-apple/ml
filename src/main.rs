mod lib;

use std::time::{SystemTime, UNIX_EPOCH};

use plotters::prelude::{BitMapBackend, RED, DrawingBackend};

use lib::Rng;

fn main() {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let mut rng = Rng::new(now.as_secs());

    for _ in 0..100 {
        let random = rng.rand();
        println!("{random}");
    }

    // Create a 800*600 bitmap and start drawing
    let mut backend = BitMapBackend::new("test/1.png", (300, 200));
    // And if we want SVG backend
    // let backend = SVGBackend::new("output.svg", (800, 600));
    backend.draw_rect((50,50), (200, 150), &RED, true).unwrap();
    backend.present().unwrap();
}