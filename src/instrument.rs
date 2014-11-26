use super::units::{
    Mass,
    Length,
    Velocity,
    Accel,
    Force,
    Stiffness,
    Damping
};

// Opaque handles on components of an Instrument.
// Privately these are just indexes into Instrument's vectors.
pub struct Particle(uint);
pub struct Spring(uint);

// Specification of a particle
pub struct ParticleSpec {
    pub mass: Mass,
    pub position: Length
}

// Specification of a spring
pub struct SpringSpec {
    pub p_0: Particle,
    pub p_1: Particle,
    pub natural_length: Length,
    pub stiffness: Stiffness,
    pub damping: Damping,
    pub one_sided: bool
}

pub struct Instrument {
    particles: Vec<ParticleSpec>,
    springs: Vec<SpringSpec>,
    earths: Vec<Particle>
}

impl Instrument {
    pub fn new() -> Instrument {
        Instrument{
            particles: Vec::new(),
            springs: Vec::new(),
            earths: Vec::new()
        }
    }

    pub fn add_particle(&mut self, mass: Mass, position: Length) -> Particle {
        self.particles.push(ParticleSpec{ mass: mass, position: position });
        Particle(self.particles.len() - 1)
    }

    // With n=3: (p_0) -- (1) -- (2) -- (p_n)
    pub fn add_chain(&mut self, p_0: Particle, p_n: Particle, n: uint, mass: Mass, stiffness: Stiffness, damping: Damping) {
        let mut p_prev = p_0;
        for _ in range(1, n) {
            let p_i = self.add_particle(mass, Length(0.0));
            self.add_spring(p_prev, p_i, Length(0.0), stiffness, damping, false);
            p_prev = p_i;
        }
        self.add_spring(p_prev, p_n, Length(0.0), stiffness, damping, false);
    }

    pub fn add_spring(&mut self,
                        p_0: Particle,
                        p_1: Particle,
                        natural_length: Length,
                        stiffness: Stiffness,
                        damping: Damping,
                        one_sided: bool) -> Spring {
        self.springs.push(SpringSpec{
            p_0: p_0,
            p_1: p_1,
            natural_length: natural_length,
            stiffness: stiffness,
            damping: damping,
            one_sided: one_sided
        });
        Spring(self.springs.len() - 1)
    }

    pub fn earth(&mut self, p: Particle) {
        self.earths.push(p);
    }
}

#[deriving(Clone, Show)]
pub struct ParticleState {
    pub position: Length,
    pub velocity: Velocity,
}

impl ParticleState {
    fn new() -> ParticleState {
        ParticleState { position: Length(0.0), velocity: Velocity(0.0) }
    }
}

impl Add<ParticleState, ParticleState> for ParticleState {
    fn add(&self, rhs: &ParticleState) -> ParticleState {
        ParticleState {
            position: self.position + rhs.position,
            velocity: self.velocity + rhs.velocity
        }
    }
}

impl Mul<f64, ParticleState> for ParticleState {
    fn mul(&self, &rhs: &f64) -> ParticleState {
        ParticleState {
            position: self.position * rhs,
            velocity: self.velocity * rhs
        }
    }
}

#[deriving(Clone)]
pub struct InstrumentState<'a> {
    particle_states: Vec<ParticleState>,
    instrument: &'a Instrument
}

impl<'a> InstrumentState<'a> {
    pub fn new(instrument: &'a Instrument) -> InstrumentState<'a> {
        InstrumentState {
            particle_states: Vec::from_elem(instrument.particles.len(), ParticleState::new()),
            instrument: instrument
        }
    }

    pub fn particle_state(&self, Particle(index): Particle) -> ParticleState {
        self.particle_states[index]
    }

    pub fn trigger_hammer(&mut self,
                            Particle(hammer_index): Particle,
                            Particle(target_index): Particle,
                            velocity: Velocity) {
        self.particle_states[hammer_index] = ParticleState {
            position: self.particle_states[target_index].position,
            velocity: velocity
        };
    }
}

impl<'a> Add<InstrumentState<'a>, InstrumentState<'a>> for InstrumentState<'a> {
    fn add(&self, rhs: &InstrumentState) -> InstrumentState<'a> {
        assert_eq!(self.particle_states.len(), rhs.particle_states.len());
        InstrumentState {
            particle_states: self.particle_states.iter().enumerate().map(|(index, &left)| {
                left + rhs.particle_states[index]
            }).collect(),
            instrument: self.instrument
        }
    }
}

impl<'a> Mul<f64, InstrumentState<'a>> for InstrumentState<'a> {
    fn mul(&self, &rhs: &f64) -> InstrumentState<'a> {
        InstrumentState {
            particle_states: self.particle_states.iter().map(|&left| {
                left * rhs
            }).collect(),
            instrument: self.instrument
        }
    }
}

impl<'a> ::runge_kutta::State for InstrumentState<'a> {
    // Calculate the time-derivative of the instrument's state.
    // This returns an InstrumentState with each ParticleState having its
    // velocity in particle.position and its acceleration in particle.velocity.
    fn f(&self, _: f64) -> InstrumentState<'a> {

        // Calculate the force to be applied by each spring
        // and sum the forces on each particle.
        let particle_count = self.instrument.particles.len();
        let mut forces = Vec::from_elem(particle_count, Force(0.0));
        for spring in self.instrument.springs.iter() {
            let Particle(p_0_index) = spring.p_0;
            let Particle(p_1_index) = spring.p_1;
            let p_0 = self.particle_states[p_0_index];
            let p_1 = self.particle_states[p_1_index];
            // negative strain is in compression, positive is in tension
            let strain = p_1.position - spring.natural_length - p_0.position;
            let dstraindt = p_1.velocity - p_0.velocity;
            let Length(strain_val) = strain;
            if spring.one_sided && strain_val > 0.0 {
                // no force
            } else {
                let force = spring.stiffness * strain + spring.damping * dstraindt;
                forces[p_0_index] = forces[p_0_index] + force;
                forces[p_1_index] = forces[p_1_index] - force;
            }
        }

        // Zero the forces on the earthed particles so they don't move
        for &Particle(index) in self.instrument.earths.iter() {
            forces[index] = Force(0.0);
        }

        // Return a time-derivative of the particles' states.
        InstrumentState {
            particle_states: self.particle_states.iter().enumerate().map(|(index, &particle)| {
                // Returning a time-derivative as an "InstrumentState" means the units mismatch.
                // Otherwise the RK functions would have to deal with another type.
                let Velocity(velocity) = particle.velocity;
                let Accel(accel) = forces[index] / self.instrument.particles[index].mass;
                ParticleState {
                    position: Length(velocity),
                    velocity: Velocity(accel)
                }
            }).collect(),
            instrument: self.instrument
        }
    }
}
