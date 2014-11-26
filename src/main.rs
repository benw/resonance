#![feature(macro_rules)]

extern crate openal;

use std::io::File;
use runge_kutta::step_rk4;
use units::{
    Mass,
    Length,
    Velocity,
    Stiffness,
    Damping
};
use instrument::{Particle, Instrument, InstrumentState};
use std::i16;
use openal::{al, alc};
use std::io::timer::sleep;
use std::time::duration::Duration;

mod runge_kutta;
mod units;
mod instrument;

#[allow(dead_code)]
fn make_simple_instrument(mut instrument: Instrument) -> (Instrument, Particle) {
    let p_0 = instrument.add_particle(Mass(1.0), Length(0.0));
    let p_1 = instrument.add_particle(Mass(1.0e-2), Length(0.0));
    instrument.add_spring(p_0, p_1, Length(1.0), Stiffness(1.0e3), Damping(1e-1), false);
    instrument.earth(p_0);
    (instrument, p_1)
}

fn make_string() -> (Instrument, Particle, Particle, Particle) {
    let hammer_mass = Mass(0.05);
    let hammer_stiffness = Stiffness(1e3);
    let string_mass = Mass(1e-2);
    let string_stiffness = Stiffness(3e5);
    let string_damping = Damping(0.5);
    let air_damping = Damping(0.3);

    let mut instrument = Instrument::new();
    let p_earth = instrument.add_particle(string_mass, Length(0.0));
    let p_target = instrument.add_particle(string_mass, Length(0.0));
    let p_pickup = instrument.add_particle(string_mass, Length(0.0));
    let p_hammer = instrument.add_particle(hammer_mass, Length(0.0));
    instrument.earth(p_earth);
    instrument.add_chain(p_earth,  p_target, 15, string_mass, string_stiffness, string_damping);
    instrument.add_chain(p_target, p_pickup, 15, string_mass, string_stiffness, string_damping);
    instrument.add_chain(p_pickup, p_earth,  15, string_mass, string_stiffness, string_damping);
    instrument.add_spring(p_hammer, p_target, Length(0.0), hammer_stiffness, Damping(0.0), true);
    instrument.add_spring(p_pickup, p_earth, Length(0.0), Stiffness(0.0), air_damping, false);
    (instrument, p_hammer, p_target, p_pickup)
}

fn slice_as_bytes<'a, T>(slice: &'a [T]) -> &'a [u8] {
    unsafe {
        std::mem::transmute(std::raw::Slice {
            data: slice.as_ptr(),
            len: slice.len() * std::mem::size_of::<T>()
        })
    }
}

fn main() {
    let hammer_velocity =  Velocity(1e1);
    let (instrument, p_hammer, p_target, p_pickup) = make_string();

    let mut state = InstrumentState::new(&instrument);
    state.trigger_hammer(p_hammer, p_target, hammer_velocity);

    let sample_freq = 44100.0;
    let duration = 10.0;
    let num_samples = (sample_freq * duration) as uint;
    let dt = 1.0 / sample_freq;
    let mut peak = 0.0f64;

    println!("Generating");

    let amplitudes = Vec::from_fn(num_samples, |i| {
        let t = (i as f64) * dt;
        state = step_rk4(t, dt, &state);
        let Velocity(amplitude) = state.particle_state(p_pickup).velocity;
        if amplitude > peak {
            peak = amplitude;
        }
        if -amplitude > peak {
            peak = -amplitude;
        }
        amplitude
    });
    if peak == 0.0 {
        // avoid div-by-zero for all-zero waveform
        peak = 1.0;
    }
    let samples: Vec<i16> = amplitudes.iter().map(|amplitude| {
        (*amplitude / peak * (i16::MAX - 1) as f64) as i16
    }).collect();

    let device = alc::Device::open(None).expect("Could not open device");
    let ctx = device.create_context(&[]).expect("Could not create context");
    ctx.make_current();

    let buffer = al::Buffer::gen();
    let source = al::Source::gen();

    println!("Writing");

    {
        let mut f = File::create(&Path::new("sound.raw"));
        f.write(slice_as_bytes(samples.as_slice()));
    }

    unsafe { buffer.buffer_data(al::FormatMono16, samples.as_slice(), sample_freq as al::ALsizei) };

    println!("Playing");

    source.queue_buffer(&buffer);
    source.play();
    while source.is_playing() {
        sleep(Duration::milliseconds((duration * 100.0) as i64));
    }

    ctx.destroy();
    device.close().ok().expect("Unable to close device");
}
