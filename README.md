# solve_omp_opt: High-Performance Parallel Binary Linear Solver

`solve_omp_opt` is a high-performance constraint satisfaction solver implemented in C++23. It is engineered to find all binary solutions ($x_i \in \{0, 1\}$) for systems of linear equations of the form:

$$\sum a_{ij} x_j = b_i$$

The solver achieves millions of solutions per second through a combination of cache-aware data structures, aggressive mathematical pruning, and hybrid multi-core parallelism.

## 1. Mathematical Foundries & Optimizations

### 1.1. Constraint Propagation
For every equation, the solver tracks the "Target" ($T_r$: required sum) and the "Remainder" ($P_r, N_r$: potential positive/negative contribution from unassigned variables). A branch is pruned if $T_r$ falls outside the range $[N_r, P_r]$.

### 1.2. Gap Pruning (Hot-Loop Optimization)
The solver implements a specialized "Gap Pruning" check. By tracking the maximum absolute coefficient $|c|_{max}$ for each row, it can bypass expensive variable scans if:
$$N_r + |c|_{max} - 1 < T_r < P_r - |c|_{max} + 1$$
In this state, it is mathematically impossible for any single variable to be "forced," allowing the solver to skip the row and significantly speed up propagation.

## 2. System Architecture

### 2.1. Flattened Memory Layout
To maximize CPU cache efficiency, the solver uses a flattened memory model. All row data (variables and coefficients) are stored in contiguous arrays, eliminating pointer chasing and ensuring linear memory access patterns during constraint checks.

### 2.2. Trail-Based State Management
Instead of expensive state copying during search, the solver uses a **Backtracking Trail**. It records changes to row targets and remainders on a stack and restores them in $O(1)$ time during backtracking, resulting in a zero-allocation hot loop.

### 2.3. Hybrid OpenMP Parallelism
- **Task Phase (Breadth)**: Top-level search nodes are managed as OpenMP tasks, providing dynamic load balancing across all available cores.
- **Sequential Phase (Depth)**: Once the `--spawn-depth` is reached, each core executes a high-speed recursive DFS.

## 3. Storage & Compression

The solver supports an optional, high-performance compressed storage format using **Zstandard**.

### 3.1. File Formats
- **Standard Binary**: Raw bit-packed vectors (1 bit per variable, 64-bit aligned).
- **Compressed (Zstd)**: Batch-compressed blocks (1024 solutions per block). Files are marked with the `ZSD1` magic header and offer **5x–10x space reduction**.

### 3.2. macOS Installation
```bash
# Install Zstd library
brew install zstd

# Install Python support for decoding
python3 -m pip install zstandard
```

## 4. Building & Usage

### 4.1. Compilation
**Standard Build:**
```bash
c++ -fopenmp -O3 -march=native --std=c++23 solve_omp_opt.cpp -o solve_omp_opt
```

**Build with Zstd Support:**
```bash
# Note: Include/Library paths may vary (/opt/homebrew for Apple Silicon, /usr/local for Intel)
c++ -fopenmp -O3 -march=native --std=c++23 -DUSE_ZSTD \
    -I/opt/homebrew/include -L/opt/homebrew/lib -lzstd \
    solve_omp_opt.cpp -o solve_omp_opt
```

### 4.2. Execution
```bash
./solve_omp_opt dominik_equations.txt --threads 12 --spawn-depth 24 --count-only
```

### 4.3. Decoding Solutions
The included `decode_solutions.py` script automatically detects and handles both compressed and uncompressed files:
```bash
./decode_solutions.py <solutions_dir_or_file> <nvars>
```

## 5. Performance Metrics
On a 12-core system with 510 variables and ~400 equations:
- **Baseline Throughput**: ~4.1 Million solutions/second.
- **Compressed Throughput**: ~3.1 Million solutions/second.
- **Storage Efficiency**: Reduced 147M solutions from **8.8 GB** to **1.3 GB**.
