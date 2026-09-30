# CS3.406 - PAA Assignment 1 Submission

This folder contains the codebase, experimental results, analytical scripts, generated plots, and the final technical report for the CUDA SGEMM Optimization assignment.

## 📂 Folder Structure

- **`report.pdf`**: The final comprehensive performance analysis report.
- **`SGEMM_T4_Colab.ipynb`**: The Jupyter Notebook used to run and collect all experiments on the Tesla T4 GPU (via Google Colab).
- **`src/`**: The CUDA C++ source code. Contains all 10 kernel versions (`kernels.cuh`) and the test harness (`runner.cu`).
- **`scripts/`**: Shell scripts to automate different experimental sweeps locally.
- **`analysis/`**: Python scripts to aggregate data across GPUs and generate plots.
- **`results/`**: Contains the raw text logs and NCU CSVs from both GPUs (`rtx3050/` and `t4/`).
  - **`results/combined/`**: Aggregated CSVs created by the parsers.
  - **`results/plots/`**: All 15 final generated PNG plots used in the report (rooflines, sweeps, hardware metrics).
- **`Makefile`** & **`CMakeLists.txt`**: Build configurations.

## 🛠️ Environment & Setup

We conducted the experiments across two architectures:
1. **NVIDIA RTX 3050 (Ampere, sm_86)**: Run locally via WSL Ubuntu 22.04 using native CUDA toolkit (`nvcc`) and Nsight Compute (`ncu`).
2. **NVIDIA Tesla T4 (Turing, sm_75)**: Run on Google Colab using the provided notebook.

### Building the CUDA code (Local)
To compile the kernels locally (requires `nvcc`), simply run:
```bash
make
```
*(This will compile the `sgemm` executable into a `build/` directory).*

## 🚀 How the Experiments Were Run

### 1. Tesla T4 (Colab)
Upload `SGEMM_T4_Colab.ipynb` to Google Colab, select a T4 GPU runtime, and run all cells. The notebook automatically compiles the code, runs the benchmark sweeps (Square, Non-Square, Nsight Compute, Parameter Sweep), and saves the results to the `results/t4/` folder.

### 2. RTX 3050 (Local / WSL)
The local experiments were orchestrated using the shell scripts in `scripts/`:

*   **Experiment 1, 2 & 3 (Performance Sweeps):**
    ```bash
    ./scripts/run_all.sh
    ```
    This runs the square matrix sweep (E1), non-square shapes (E2), and non-power-of-two finesweep (E3), dumping raw text output to `results/rtx3050/runtime/`.

*   **Experiment 4 (Hardware Metrics via Nsight Compute):**
    ```bash
    ./scripts/profile.sh
    ```
    This profiles L1/L2 caches, DRAM, register pressure, and coalescing for all kernels at N=4096, saving to `results/rtx3050/ncu/metrics.csv`. *(Requires admin/root privileges for Nsight counters).*

*   **Experiment 5 (Parameter Sweep):**
    ```bash
    ./scripts/param_sweep.sh
    ```
    This runs Autotuned Kernel 9 across a grid of block and thread tiling configurations.

## 📊 Data Processing & Plotting

After execution, the raw text outputs from both GPUs were processed into combined, easy-to-read CSV files and visualised using Python. 

To reproduce the plots locally (requires `pandas`, `matplotlib`, `seaborn`):

1. **Aggregate Runtime Data**:
   ```bash
   python analysis/collect_runtime.py
   ```
   *Parses the raw benchmark logs from `results/<gpu>/runtime/` into a single `results/combined/runtime_all.csv`.*

2. **Aggregate Hardware Metrics**:
   ```bash
   python analysis/parse_ncu.py
   ```
   *Parses and normalizes the raw Nsight Compute CSVs into `results/combined/metrics_all.csv`.*

3. **Generate Standard Plots**:
   ```bash
   python analysis/plots.py
   ```
   *Reads the combined CSVs and generates the bar charts, cross-GPU comparisons, metric heatmaps, and sawtooth line plots.*

4. **Generate Roofline Models**:
   ```bash
   python analysis/roofline.py
   ```
   *Calculates actual Arithmetic Intensity (AI) using the profiled DRAM bytes and generates roofline plots for both architectures.*

---
*All generated visualisations output automatically to the `results/plots/` directory.*
