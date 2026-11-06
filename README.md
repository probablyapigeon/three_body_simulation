# Audio-Reactive Phase-Space Engine: Three-Body Trajectory Toy Model

![Stable Three-Body Orbits](stable_three_body.gif)

## Abstract & System Scope
This repository houses a self-contained, audio-reactive visualization engine designed to track localized matrix trajectories within a bounded phase space, with complete documentation available in the repository files.

## System Pipeline Requirements
The script runs natively on stable Python 3.12+ environments without requiring external C-compilers or system level audio drivers.

```powershell
# Setup and activate your environment
python -m venv stable_3body_env
.\stable_quantum_env\Scripts\Activate.ps1

# Install required mathematical and signal processing libraries
pip install torch networkx numpy matplotlib miniaudio
```

## Running the Simulation

```powershell
python three_body_simulation.py
```
