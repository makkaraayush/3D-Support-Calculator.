# 3D Support Calculator & Self-Learning AI Optimizer

A physical simulation engine and continuous self-learning neural surrogate optimizer that calculates high-efficiency, easily detachable tree support structures for FDM 3D printers, specifically calibrated for the Bambu Lab A1.

---

## Why I Built This

I recently got my very first 3D printer, a Bambu Lab A1. Out of the box, the default slicer supports and standard tree supports worked pretty well for basic prints. But as I started printing more complex mechanical parts, models with delicate overhangs, and horizontal holes, I ran into the usual headaches:

1. Standard grid supports waste a massive amount of filament and are practically fused to the model surface. Removing them often leaves ugly scarring or requires pliers and flush cutters that risk snapping thin walls.
2. Even regular tree supports sometimes generate unnecessary branches inside internal bore holes where they are almost impossible to reach and extract.
3. Pre-set slicer profiles rely on fixed static parameters. They do not actively test and simulate physics across alternative branch topologies.

As an engineering student, I did not want to limit myself to fixed preset defaults. I wanted to see if I could write a program that pushes the boundaries of support physics: a system that tests possibilities continuously, evaluates beam buckling, overhang sag, and interfacial release forces, and uses a custom local AI model that trains itself online. The longer you leave it running, the more possibilities it explores and the more intelligent it becomes at finding Pareto-optimal support structures.

I also made sure this runs smoothly on low-end hardware. My own daily workstation is an Intel Core i5-2400s with an NVIDIA Quadro K2200 GPU, so the entire codebase is written in pure, vectorized NumPy and SciPy with zero heavy multi-gigabyte machine learning bloat. Forward passes take less than 0.1 milliseconds, making continuous background training fast and responsive even on older 4-core desktop CPUs.

---

## Core Capabilities

- Physics-First Calculation: Precomputes established FDM fluid/thermal relationships so low-power CPUs do not waste computation on known physics. Includes Euler-Bernoulli cantilever deflection, Euler column buckling safety factors, bed adhesion overturning moments, and exponential contact layer shear release curves.
- Multi-Objective Optimization: Automatically balances five competing print objectives:
  - Overhang coverage (zero droop / failure prevention)
  - Filament weight minimization (up to 40% to 70% savings compared to standard grid supports)
  - Breakaway ease (calculates exact peel force in Newtons for 1-snap detachment)
  - Structural buckling stability under dynamic nozzle drag
  - Total print time and travel move minimization
- Online Self-Learning Neural Surrogate: A lightweight neural network with Adam optimization that learns from physical simulations in real-time. Candidate support topologies are pre-screened by the surrogate before physical evaluation, speeding up exploration by orders of magnitude.
- Hole & Internal Bore Intelligence: Classifies horizontal holes into micro-arches (which self-support without plastic), medium bores (sacrificial bridge roofs), and large cylinders (single-point cantilever plugs that pull out cleanly in one piece).
- Full 3MF & Bambu Studio Integration: Upload raw STL, OBJ, or full sliced `.3mf` Bambu Studio projects. The engine extracts plate geometries and slicing configs, optimizes the supports, and exports ready-to-print `.3mf` packages or combined `.stl` files.
- Full-Stack Interactive Web Dashboard: Includes a real-time Three.js 3D viewport with Bambu A1 bed dimensions, overhang angle heatmaps, live background training metrics, and comparison tables.

---

## System Architecture

```
3D-Support-Calculator/
├── src/
│   ├── physics/
│   │   ├── constants.py         # Material database (PLA, PETG, ABS, TPU) & Bambu A1 specs
│   │   ├── beam_mechanics.py    # Buckling, beam deflection, and breakaway shear physics
│   │   └── precalculated.py     # Precalculated air gap tables, bridge limits, hole rules
│   ├── geometry/
│   │   ├── mesh.py              # Pure NumPy triangle mesh structure with divergence volume
│   │   ├── stl_parser.py        # High-speed binary & ASCII STL reader/writer
│   │   ├── obj_parser.py        # Wavefront OBJ parser
│   │   ├── bambu_3mf.py         # 3MF package reader, XML parser, and Bambu Studio exporter
│   │   └── overhang_analyzer.py # Vectorized face normal analysis & contact clustering
│   ├── generator/
│   │   ├── tree_graph.py        # Hierarchical graph of trunks, branches, and tips
│   │   ├── path_routing.py      # Collision-avoiding 3D branch routing to bed (Z=0)
│   │   ├── interface_layers.py  # Conical pencil tips, air gap spacing, and release rating
│   │   └── mesh_builder.py      # Translates graph into solid 3D manifold mesh
│   ├── simulator/
│   │   └── physics_evaluator.py # Evaluates coverage, mass, breakaway force, stability, time
│   ├── ai/
│   │   ├── surrogate_model.py   # Pure NumPy neural surrogate model with Adam optimizer
│   │   ├── evolutionary_search.py # Genetic parameter mutation and Pareto ranking
│   │   └── continuous_engine.py # Background thread daemon with persistent replay buffer
│   ├── server/
│   │   └── app.py               # FastAPI backend with REST APIs and SSE live streaming
│   └── web/
│       ├── index.html           # Full-stack dashboard interface
│       ├── styles.css           # Clean dark theme layout and gauge components
│       ├── viewer.js            # Three.js 3D viewport with OrbitControls and heatmaps
│       └── app.js               # Frontend controller, slider bindings, and SSE client
├── tests/                       # Complete automated unit test suite
├── data/
│   └── benchmark_models/        # Overhang torture, arch bridge, and hole test STL files
├── run.py                       # Launcher for web dashboard or headless CLI
└── requirements.txt             # Lightweight dependencies
```

---

## Physics & Calculation Fundamentals

### 1. Overhang Overlap Ratio
An extruded perimeter bead of width $w$ and layer height $h$ printed at an overhang angle $\theta$ relative to vertical has an overlap ratio with the preceding layer:

$$R = 1 - \frac{h}{w} \cot(\theta)$$

When $R < 0.28$, surface tension and gravity overcome cooling adhesion, causing filament sag and curling. The Bambu Lab A1 dual-duct part cooling fan boosts safe unsupported angles up to approximately 58 degrees for standard PLA at 0.20 mm layer height.

### 2. Support Column Euler Buckling
Tree support branches descending to the build bed must resist lateral nozzle drag forces ($F_{\text{drag}} \approx 0.85\text{ N}$). The critical buckling threshold is calculated via Euler's formula:

$$P_{\text{cr}} = \frac{\pi^2 E I}{(K L)^2}$$

Where:
- $E$ is the material elastic modulus (3.5 GPa for PLA, 2.1 GPa for PETG).
- $I = \frac{\pi d^4}{64}$ is the second moment of area for a circular column.
- $K = 2.0$ represents a fixed-base cantilever column.

If the safety factor $P_{\text{cr}} / F_{\text{drag}} < 2.5$, the optimizer flags instability and automatically increases trunk taper or adds convergence branches.

### 3. Contact Interface Breakaway Force
Effortless post-print support removal requires minimizing the mechanical peel force without causing premature detachment during printing:

$$F_{\text{breakaway}} = A_{\text{contact}} \cdot \tau_{\text{interfacial}} \cdot \exp\left(-\frac{z_{\text{gap}}}{0.65 \cdot h_{\text{layer}}}\right)$$

- At $z_{\text{gap}} = 0.0\text{ mm}$, the support fuses with the part ($\tau \approx 1.2\text{ MPa}$).
- At $z_{\text{gap}} = 0.22\text{ mm}$ (for 0.20 mm PLA layers), individual tip detachment force drops to approximately 0.05 - 0.20 N per tip. This allows the entire tree support structure to break away cleanly with a single manual pull, leaving no white stress marks or surface blemishes.

---

## Installation & Quick Start

### Prerequisites
- Python 3.9 or higher
- Standard pip package manager

### 1. Clone the Repository
```bash
git clone https://github.com/makkaraayush/3D-Support-Calculator..git
cd "Support Calculator"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Web Dashboard
```bash
python run.py
```
This automatically initializes the server and opens your browser at:
```
http://127.0.0.1:8000
```

### 4. Run in Headless CLI Mode (for automated benchmarking)
```bash
python run.py --cli --model data/benchmark_models/overhang_torture.stl --iterations 25 --material PLA
```

---

## Running Automated Tests

A comprehensive unit test suite is included in `tests/`:

```bash
# Run all test suites
python -m unittest discover tests

# Or run individual modules
python -m unittest tests/test_physics.py
python -m unittest tests/test_mesh_parser.py
python -m unittest tests/test_tree_generator.py
python -m unittest tests/test_ai_optimizer.py
```

All tests execute in under 4 seconds on typical budget desktop CPUs.

---

## Bambu Studio / OrcaSlicer Workflow

1. Open the web interface at `http://127.0.0.1:8000`.
2. Drag and drop your `.stl`, `.obj`, or `.3mf` project onto the viewport.
3. Select your material (PLA, PETG, ABS, TPU) and desired layer height.
4. Click **Start Self-Learning Search** to let the background AI engine test branch angles, merge radii, and air gaps.
5. Once you are satisfied with the filament savings and removal ease rating, click **Export Bambu 3MF**.
6. Open the exported `.3mf` directly in Bambu Studio or OrcaSlicer, verify the sliced preview, and send it to your Bambu Lab A1 printer.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
