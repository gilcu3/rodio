//! Demonstrates pitch-preserving vs pitch-shifting playback speed.
//!
//! Run with: `cargo run --example speed_control`
//!
//! The same passage is replayed four times so you can compare:
//!   1. normal speed,
//!   2. 1.5x faster, pitch preserved (the default),
//!   3. 0.75x slower, pitch preserved,
//!   4. 1.5x faster with pitch preservation turned off (pitch rises, the old
//!      `set_speed` behaviour).

use std::error::Error;
use std::thread::sleep;
use std::time::Duration;

fn main() -> Result<(), Box<dyn Error>> {
    let stream_handle = rodio::DeviceSinkBuilder::open_default_sink()?;
    let player = rodio::Player::connect_new(stream_handle.mixer());

    let file = std::fs::File::open("assets/music.ogg")?;
    player.append(rodio::Decoder::try_from(file)?);

    // How long to listen to each setting before moving on.
    let segment = Duration::from_secs(4);

    println!("1) normal speed (1.0x)");
    player.set_speed(1.0);
    sleep(segment);

    println!("2) 1.5x faster, pitch preserved (default)");
    player.try_seek(Duration::ZERO)?;
    player.set_speed(1.5);
    sleep(segment);

    println!("3) 0.75x slower, pitch preserved");
    player.try_seek(Duration::ZERO)?;
    player.set_speed(0.75);
    sleep(segment);

    println!("4) 1.5x faster, pitch NOT preserved (pitch rises)");
    player.try_seek(Duration::ZERO)?;
    player.set_preserve_pitch(false);
    player.set_speed(1.5);
    sleep(segment);

    println!("done");
    Ok(())
}
