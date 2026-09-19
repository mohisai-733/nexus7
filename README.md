# Adaptive Resource Allocation in Multiprogramming Systems

NEXUS-7 simulates OS-level adaptive resource scheduling with real-time telemetry, alerting, and interactive controls.

## Overview

This project demonstrates how an operating system can dynamically allocate CPU and memory across competing processes under changing workload pressure.

The repository includes:
- A classic simulation script (`os project file.py`)
- A real-time web engine + dashboard (`nexus7.py`, `nexus7_dashboard.html`, `nexus7_landing.html`)
- A terminal-based real-time simulator (`nexus7_terminal.py`)

## Current Project Structure

```
.
├── nexus7.py                 # WebSocket engine + Flask server + scheduler suite
├── nexus7_dashboard.html     # Real-time visual dashboard UI
├── nexus7_landing.html       # Landing page for the project
├── nexus7_terminal.py        # Interactive terminal simulator
├── os project file.py        # Original standalone simulation
├── nexus7_requirements.txt   # Python dependencies
├── README.md                 # Documentation
└── SUBMISSION_NOTES.md       # Submission context
```

## Features

- Adaptive multi-process CPU/MEM allocation with weighted fairness
- Multiple schedulers: RR, Priority, SJF, MLQ, MLFQ, and Nexus Adaptive AI
- EWMA-based bottleneck detection (`none`, `cpu`, `memory`, `cpu+mem`)
- Collision detection for contention spikes (CPU, memory, dual)
- Real-time alerts in both web and terminal modes
- Alert sensitivity profiles (`sensitive`, `balanced`, `severe`)
- Manual request override mode for controlled experiments
- SQLite telemetry logging and linear-regression trend prediction
- Ghost replay and battle mode (web engine)

## Tech Stack

| Component | Technology |
| --- | --- |
| Language | Python 3.9+ |
| Backend | Flask, Flask-SocketIO |
| UI | HTML/CSS/JavaScript + Socket.IO |
| Data/ML | Pandas, NumPy, scikit-learn |
| System Metrics | psutil |
| Persistence | SQLite |

## Setup

```bash
git clone https://github.com/Eswar-2006/Adaptive-Resource-Allocation-in-Multiprogramming-Systems.git
cd Adaptive-Resource-Allocation-in-Multiprogramming-Systems
pip install -r nexus7_requirements.txt
```

## Run Modes

### 1) Web Engine + Dashboard

```bash
python nexus7.py
```

Open:
- `http://localhost:7800/landing`
- `http://localhost:7800/dashboard`

### 2) Terminal Real-Time Simulator

```bash
python nexus7_terminal.py
```

Terminal commands:
- `help`
- `mode manual` / `mode auto`
- `set <pid> <cpu%> <memMB>`
- `del <pid>`
- `clear`
- `quit`

Example:

```text
mode manual
set 1 30 400
set 3 15 220
```

## Alerting Model

Alerts are evaluated every tick from EWMA pressure and under-fulfillment contention.

- Bottleneck alerts:
	- `cpu`, `memory`, or `cpu+mem`
- Collision alerts:
	- `cpu`, `memory`, or `cpu+mem`
	- based on concurrent under-fulfilled contenders

In terminal mode, collision alerts are shown as:
- a dedicated collision banner line
- per-tick alert message entries in the `Messages` panel

## Alert Sensitivity Profiles

You can tune alert noise/sensitivity without code edits.

Environment variable:
- `NEXUS_ALERT_PROFILE`: applies to web engine and also terminal (default fallback)
- `NEXUS_TERMINAL_ALERT_PROFILE`: terminal-only override

Accepted values:
- `sensitive` (earlier alerts)
- `balanced` (default)
- `severe` (only stronger spikes)

PowerShell examples:

```powershell
$env:NEXUS_ALERT_PROFILE = "sensitive"
python nexus7.py
```

```powershell
$env:NEXUS_TERMINAL_ALERT_PROFILE = "severe"
python nexus7_terminal.py
```

## Notes

- Telemetry is written each tick to SQLite.
- ML prediction appears after enough datapoints are collected.
- Manual mode uses only user-defined process requests; unspecified PIDs request `0`.