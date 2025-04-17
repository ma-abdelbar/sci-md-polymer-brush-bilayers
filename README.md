# sci-md-polymer-brush-bilayers

**Simulating Polymer Brush Bilayers under Compression and Shear using Coarse-Grained Molecular Dynamics (CGMD)**

This repository contains the complete simulation and analysis framework developed during the PhD thesis of **Mohamed A. Abdelbar** at Imperial College London. It enables generation, execution, and analysis of large-scale molecular dynamics simulations of polymer brush bilayers under confinement and shear.

> 📄 **Reference**  
> Abdelbar, M. A. *The Effect of Chain Architecture and Brush Topology on the Tribology of Polymer Brushes using Coarse-grained Molecular Dynamics.*  
> PhD Thesis, Imperial College London, 2024.  
> **Access under embargo**. This thesis will be publicly available via the Imperial College London Spiral repository once the embargo lifts.
>
> This work was funded by the **EPSRC** through the **TSM CDT** (EP/L015579/1).
> License: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
---

## ✨ Key Features

- Full LAMMPS input pipeline to simulate:
  - Equilibrium structure of polymer brushes
  - Compression response of bilayers
  - Shear dynamics and tribological properties
- Supports **linear**, **cyclic**, **loop**, and **bottlebrush** chain architectures
- Supports implicit or explicit solvents.
- Easily extensible to new chemistries, solvents, or topologies
- Includes:
  - Parameter sweep generation (`BuildSims.py`)
  - HPC execution logic (`RunSims.py`, PBS scripts)
  - Postprocessing to CSV (`post.py`)
  - Aggregated analysis via pickled numpy arrays
  - Visualization of results and snapshots using Ovito

---

## 🗂 Folder Structure

```bash
sci-md-polymer-brush-bilayers/
│
├── simulation_scripts/       # Seed folder copied per study (e.g., Linear, Loop, BB)
│   ├── PBB-ECS/              # Core simulation structure for 3-stage run (Eq→Cp→Sh)
│   ├── main.in               # Top-level LAMMPS control script
│   ├── PBB.in                # Brush bilayer construction
│   ├── ECS.in                # Main simulation protocol
│   ├── BuildSims.py          # Creates full tree of simulations
│   ├── RunSims.py            # Runs all simulations in HPC tree
│   ├── arrayJS.pbs           # PBS job script for Cx1 cluster
│   └── ...                   # Auxiliary scripts
│
├── results/
│   ├── get_data.ipynb        # Loads all results into numpy arrays + pickles
│   ├── analysis.ipynb        # Scaling laws, plotting, comparisons
│   ├── visualization.ipynb   # OVITO-based snapshot analysis
│   └── data/                 # Pickled data arrays
│
├── README.md                 # (You are here)
├── requirements.txt          # Python dependencies (minimal)
├── LICENSE                   # CC BY-NC 4.0
└── ...
```
---
## 📦 Installation

Create a virtual environment and install dependencies (for analysis notebooks):

```bash
python -m venv venv
source venv/bin/activate     # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```
---

## How to Use

### 1. Choose an Architecture

Copy `simulation_scripts/` to a new folder (e.g., `Linear/`, `Cyclic/`).  
Edit `main.in` to set:
- Chain length `N`
- Grafting density `M`
- Wall separation `D`
- Shear velocities `Vwalli`
- Topology logic (inside `PBB.in` and `BSMolBuilder.py`)

---

### 2. Build Simulation Tree

```bash
python BuildSims.py
```

This will generate:

```
Sims/N=*/M=*/X=*/D=*/PBB-ECS/
```

Each folder will contain a copy of `PBB-ECS` and be ready to run.

---

### 3. Run Simulations

```bash
python RunSims.py
```

This navigates the tree and launches simulations using:
- `main.in` (LAMMPS multi-stage control)
- `PBB.in` → generates grafted brushes
- `ECS.in` → performs equilibrium, compression, and shear
- `post.py` → extracts thermodynamic and structural data into CSV

---

### 4. Analyze and Visualize  
Use Jupyter notebooks in `results/`:

- `get_data.ipynb`: aggregates CSVs into structured arrays, pickles to disk  
- `analysis.ipynb`: plot any quantity vs any other (log, scalings, filters)  
- `visualization.ipynb`: generate Ovito visuals from simulation dumps  



---

## ⚖️ License

This work is distributed under a **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

You are free to:
- Share and adapt for non-commercial purposes
- Attribute the author clearly

For commercial use, please contact the author for permission.

---

## 🙋 Contact

**Mohamed A. Abdelbar**  
PhD in Materials Science, Imperial College London  
📧 maa4617@ic.ac.uk  
🔗 [GitHub](https://github.com/ma-abdelbar)
