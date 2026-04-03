# Cosmora

An interactive 3D solar system explorer built with React and Three.js.

**Live:** https://cosmora-mu.vercel.app  
**Demo:** https://youtu.be/gGv5bNrwf8Y

## What it does

Cosmora lets you explore the solar system in real time with free-fly camera navigation. Every planet is texture-mapped using NASA imagery and positioned using Kepler's orbital equations, meaning the planets are where they actually are for any date you pick, past or future.

You can track the ISS live, view near-Earth asteroids pulled from NASA's NeoWs API, and trigger a K-nearest-neighbours threat classifier that predicts asteroid danger level from size, velocity, and miss distance — built from scratch in vanilla JS with no ML libraries.

## Features

- Free-fly 3D camera with smooth orbit controls
- Accurate planetary positions computed from Kepler's laws for any historical or future date
- Live ISS tracking via wheretheiss.at API
- Near-Earth asteroid data from NASA NeoWs API
- KNN threat classifier built in vanilla JS
- Planetary alignment detection using angular position computation across all 8 planets

## Tech Stack

- React, Three.js, React Three Fiber
- NASA NeoWs API, wheretheiss.at API
- Deployed on Vercel

## Running locally
```bash
