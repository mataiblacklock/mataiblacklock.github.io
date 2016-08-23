---
title: "Astrodynamics in Python - Interplanetary Environment"
date: 2016-07-26 09:00:00
categories: Tutorial
tags: [Astrodynamics, Python, NumPy, Planets, Physics]
---

How will humans get to Mars? To answer this question, extensive virtual simulations need to be conducted in order to properly plan such a mission. This is not a simple task by any regard. From the perspective of dynamics, essentially, one must ask: starting from a parking orbit about Earth, how must I throttle the spacecraft's propulsion system at each moment in time to arrive in a parking orbit around Mars?

To make this question even more involved, constraints may be placed throughout the various stages of the spacecraft's journey. For instance: the spacecraft may only be able to launch during a certain time period, the duration of the journey can only be up to a certain value, the spacecraft's ability to withstand solar radiation is limited, or maybe there is a desired sequence of planet's to fly by.

The research field in which the dynamics of this question are subsumed is called *Astrodynamics*.

> Astrodynamics is the application of ballistics and celestial mechanics to the practical problems concerning the motion of rockets and other spacecraft. The motion of these objects is usually calculated from Newton's laws of motion and Newton's law of universal gravitation. It is a core discipline within space mission design and control. Celestial mechanics treats more broadly the orbital dynamics of systems under the influence of gravity, including both spacecraft and natural astronomical bodies such as star systems, planets, moons, and comets. Orbital mechanics focuses on spacecraft trajectories, including orbital maneuvers, orbit plane changes, and interplanetary transfers, and is used by mission planners to predict the results of propulsive maneuvers. General relativity is a more exact theory than Newton's laws for calculating orbits, and is sometimes necessary for greater accuracy or in high-gravity situations (such as orbits close to the Sun). - [Wikipedia](https://en.wikipedia.org/wiki/Orbital_mechanics)

Simple pen and paper mathematical analysis no longer suffices for this field of research. In fact, it never really did; the dynamics of interplanetary spaceflight are highly non-linear. Even in the days of the [Apollo Missions](https://www.linux.com/news/how-they-built-it-software-apollo-11), complex computations were done with the help of punch cards. The world of technology has come a very long way since then (although Fortran is still the [fastest](https://indico.esa.int/indico/event/111/session/32/contribution/126/material/paper/0.pdf)), and such computations have become as painless as ever. So painless, in fact, interplanetary missions can now be simulated in [Python](https://www.python.org/)

[<img src="https://upload.wikimedia.org/wikipedia/commons/c/c7/IBM_650_at_Texas_A%26M.jpg">](https://en.wikipedia.org/wiki/Punched_card_input/output)
*The Apollo Era*

## Definition of a Physical Body (Newtonian Mechanics)
In discussing how to utilize Python to conduct astrodynamical analyses it is fitting to begin at a rudimentary level, firstly directing attention to fundamental ideas.

In classical mechanics a physical body is collection of matter having properties including mass, velocity, momentum and energy. The matter exists in a volume of three-dimensional space. Following this definition the power of object-orientated programming can be exploited. In order to do this a class named `Body` is instantiated with a name, mass (`mass=None` to accommodate a negligible mass spacecraft), position, and velocity. For the body to be situated within a temporal context, it is necessary to instantiate the attribute of time. Additionally, it is convenient to keep track of all the physical bodies which have been instantiated in the simulation via `_instances=[]` and `Body._instances.append(self)`. Using the mathematical Python library [Numpy](http://www.numpy.org/), the physical body's attributes can be represented in double-precision floating-point format via `dtype=np.float64` and propagated expeditiously with [Fortran](https://gcc.gnu.org/fortran/) subroutines.

```python
class Body(object):
    # Instance record
    _instances = []
    def __init__(self, name, mass=None):
        # __base__ attribute will return the Body class
        self.__base__   = Body
        # Name of body
        self.name       = name
        # Mass of body
        self.mass       = np.float64(mass)
        # Time record
        self.times      = np.empty([1, 0], dtype=np.float64)
        # Position record
        self.positions  = np.empty([0, 3], dtype=np.float64)
        # Velocity record
        self.velocities = np.empty([0, 3], dtype=np.float64)
        # Append to _instances
        Body._instances.append(self)
        return None
```

Once the attributes innate to a physical body's abstract definition have been defined, more specific classifications can be made using class inheritance, in which `Body.__init__(self, name, mass)` is invoked in the descendant class's `__init__` function.

## Celestial Bodies
As interplanetary spaceflight is mainly gravitationally influenced, it is paramount to firstly instantiate the massive bodies that characterize the interplanetary environment that is the solar system.

> An astronomical object or celestial object is a naturally occurring physical entity, association, or structure that current science has demonstrated to exist in the observable universe. The term astronomical object is sometimes used interchangeably with astronomical body. Typically, an astronomical (celestial) body refers to a single, cohesive structure that is bound together by gravity (and sometimes by electromagnetism). Examples include the asteroids, moons, planets and the stars. Astronomical objects are gravitationally bound structures that are associated with a position in space, but may consist of multiple independent astronomical bodies or objects. These objects range from single planets to star clusters, nebulae or entire galaxies. A comet may be described as a body, in reference to the frozen nucleus of ice and dust, or as an object, when describing the nucleus with its diffuse coma and tail. - [Wikipedia](https://en.wikipedia.org/wiki/Astronomical_object)

The states of the various massive bodies that make up the solar system are easily attainable through [ephemerides](ftp://ssd.jpl.nasa.gov/pub/eph/planets/README.txt), which are graciously provided by [NASA's Jet Propulsion Laboratory](http://www.jpl.nasa.gov/). The main attributes of interest are the massive bodies' positions and velocities, which are easily determined for various specifications of time within [SPICE binary kernel files](ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/) as part of the [SPICE Toolkit](https://naif.jpl.nasa.gov/naif/toolkit.html). The kernels can be natively utilized in Python with the use of an open source JPL SPICE wrapper for Python, [jplephem 2.5](https://pypi.python.org/pypi/jplephem). Following the documentation, the JPL SPICE indices of the bodies must be defined. Such as it is, all the information know about the solar system's various massive bodies can be aggregated into a function that serves as a callable dictionary.

```python
def Celestial_Body_Attributes():
    '''Returns all the attributes of a specified
    celestial body. All units are SI.'''
    cba = {'Sun':     {'jplephem_index': (0, 10),
                       'mass':           1.989500e+24,
                       'diameter':       1392e+6},
           'Earth':   {'diameter':       12756000.0,
                       'jplephem_index': (3, 399),
                       'mass':           5.969999e+24},
           'Jupiter': {'diameter':       142984000.0,
                       'jplephem_index': (0, 5),
                       'mass':           1.898e+27},
           'Mars':    {'diameter':       6792000.0,
                       'jplephem_index': (0, 4),
                       'mass':           6.42e+23},
           'Mercury': {'diameter':       4879000.0,
                       'jplephem_index': (1, 199),
                       'mass':           3.3e+23},
           'Moon':    {'diameter':       3475000.0,
                       'jplephem_index': (3, 301),
                       'mass':           7.3e+22},
           'Neptune': {'diameter':       49528000.0,
                       'jplephem_index': (0, 8),
                       'mass':           1.019999e+26},
           'Pluto':   {'diameter':       2370000.0,
                       'jplephem_index': (0, 9),
                       'mass':           1.46e+22},
           'Saturn':  {'diameter':       120536000.0,
                       'jplephem_index': (0, 6),
                       'mass':           5.68e+26},
           'Uranus':  {'diameter':       51118000.0,
                       'jplephem_index': (0, 7),
                       'mass':           8.68e+25},
           'Venus':   {'diameter':       12104000.0,
                       'jplephem_index': (2, 299),
                       'mass':           4.87e+24}}
    return cba
```

Next, each celestial body's attributes can be self-assigned.

```python
def Assign_Celestial_Body_Attributes(celestial_body):
    '''This function assigns the attributes to a known celestial body.
    These attributes include the various facts of the celestial body,
    and its orbiting satellites if specified to be included.'''
    # The name of the celestial body
    name       = celestial_body.name
    # The attributes of the celestial body
    attributes = Celestial_Body_Attributes()[name]
    # Assign the factual attributes to the celestial body
    for attribute in attributes.keys():
        setattr(celestial_body, attribute, attributes[attribute])
    return None
```

And of these attributes, the body's `jplephem_index` can be used to compute its position and velocity at a given time and the fully assembled `Celestial_Body` class can finally be constructed with inheritance from the `Body` class. Also, simple computations can be performed, such as relative position and velocity `Position_and_Velocity_WRT` and updating the celestial body's state record in accordance the spacecraft's time-step `Update_Position_and_Velocity`.

```python
class Celestial_Body(Body):
    # Instance record
    _instances = []
    def __init__(self, name, mass=None, satellites=True):
        # Fundamentally initialise the celestial body
        Body.__init__(self, name, mass)
        # Assign its unique attributes
        Assign_Celestial_Body_Attributes(self)
        # If it is specified to include satellites
        Celestial_Body._instances.append(self)
        return None
    def Position_and_Velocity(self, time):
        # The JPL ephemeris index of the celestial body
        jpli       = celestial_body.jplephem_index
        # Path to ephemeris file
        path_ephem = Directory + '/Information/Celestial_Bodies/Ephemerides/de430.bsp'
        # The ephemeris kernel
        kernel     = SPK.open(path_ephem)
        # The position and velocity
        pv         = np.vstack(kernel[jpli].compute_and_differentiate(time))
        # If the ephemeris was wrt to its local barcyentre
        if not jpli[0] == 0:
            # Compute barycentric position rather
            pv     = np.add(pv, np.vstack(
                kernel[0, jpli[0]].compute_and_differentiate(time)))
        # Convert km to m
        pv[0, :]   = np.multiply(pv[0, :], 1e3)
        # Convert km/day to m/s
        pv[1, :]   = np.multiply(pv[1, :], 0.0115741)
        # Return a (2,3) numpy array
        return pv
    def Position_and_Velocity_WRT(self, body_ref, time):
        P0V0 = body_ref.Position_and_Velocity(time)
        # Measured body's barycentric position and velocity
        PV   = self.Position_and_Velocity(time)
        # Measured body's position and velocity with respect
        # to reference body
        pv   = np.subtract(PV, P0V0)
        # Returns (2,3) numpy array
        return pv
    def Update_Position_and_Velocity(self, time):
        # Barycentric position and velocity of body at time (2,3)
        pv              = self.Position_and_Velocity(time)
        # Append results to history keeping list for this body
        self.times      = np.append(self.times, time)
        self.positions  = np.vstack([self.positions, pv[0, :]])
        self.velocities = np.vstack([self.velocities, pv[1, :]])
        # Also returns barycentric position and velocty
        return None
```

Using what has just been constructed, values that may be of particular interest such as the relative position and velocity of Earth with respect to Mars at a certain Julian date, which can be used to approximate boundary conditions for interplanetary trajectory optimization, can be computed in an intuitive way.

```python
#Instantiate Earth as a massive celestial object
Earth = Celestial_Body('Earth')
#Instantiate Mars as a massive celestial object
Mars  = Celestial_Body('Mars')
#Times at which to compute position and velocity
times = [2457061.5, 2457062.5, 2457063.5, 2457064.5]
#Compute the position and velocity w.r.t. Mars
pv  = Earth.Position_and_Velocity_WRT(Mars, times[0])
```

```python
>>> pv
array([[ -3.16065185e+11,   4.67929557e+10,   2.47554111e+10],
       [ -1.59359727e+04,  -4.39852824e+04,  -1.97710792e+04]])
```

## Review
So far the abstract idea of a physical body according to classical mechanics has been programmatically implemented and augmented with the more specific classification of a celestial body. It has been shown that the state of the interplanetary environment can be determined at specified times and observed over a discretized duration. The interplanetary environment serves as the medium in which the spacecraft will travel and therfore the basis of what is necessary to answer the question presented at the beginning of this tutorial: **How will humans get to Mars?**.

The code shown here can be seen as part of a larger project [here](https://github.com/CISprague/Spacecraft_Testbed). Stay tuned for the next part of this series in which the dynamics of the Mars bound spacecraft will be described.
