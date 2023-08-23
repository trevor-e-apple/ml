mod lib;

use std::time::{SystemTime, UNIX_EPOCH};

use lib::Rng;

fn main() {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let mut rng = Rng::new(now.as_secs());

    for _ in 0..100 {
        let random = rng.rand();
        println!("{random}");
    }
}