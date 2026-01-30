## Main benchmarks

Here are the benchmarking results for the systems presented in the repository's main `README.md` for each minor `gorder` version.

### Benchmarked trajectories
- **Atomistic (CHARMM36):** 256 POPC lipids, 64 500 atoms, 10 000 trajectory frames  
- **Coarse-grained (Martini 3):** 512 POPC lipids, 16 800 beads, 10 000 trajectory frames  
- **United-atom (Berger lipids):** 256 POPC lipids, 44 300 atoms, 3 000 trajectory frames  

### System configuration
- **CPU:** 8-core Intel Core i7-11700
- **SSD:** Samsung 870 EVO
- **OS:**  GNU/Linux Mint 20.2

### Compiler versions
- `gorder` v1.0: `rustc` 1.87.0, `gcc` 9.4.0  
- `gorder` v1.1–1.3: `rustc` 1.89.0, `gcc` 9.4.0
- `gorder` v1.4: `rustc` 1.93.0, `gcc` 9.4.0

### Groan_rs versions
- `gorder` v1.0: `groan_rs` v0.10  
- `gorder` v1.1–1.3: `groan_rs` v0.11
- `gorder` v1.4: `groan_rs` v0.11

### Other information
- Benchmarked using `hyperfine` with cold cache.
- Each analysis was run 5 times.

---

### Benchmarking results

#### Atomistic system

| `gorder` version | Threads | Analysis time [s]    | Rel. to v1.0 |
|:----------------:|:-------:|:--------------------:|:------------:|
|       1.0        |    1    |   16.220 ± 0.052     |    100.0%    |
|       1.1        |    1    |   16.246 ± 0.013     |    100.2%    |
|       1.2        |    1    |   16.251 ± 0.032     |    100.2%    |
|       1.3        |    1    |   16.207 ± 0.011     |     99.9%    |
|       1.4        |    1    |   15.579 ± 0.017     |     96.0%    |
|                                                                  |
|       1.0        |    8    |   5.846 ± 0.042      |    100.0%    |
|       1.1        |    8    |   5.834 ± 0.009      |     99.8%    |
|       1.2        |    8    |   5.858 ± 0.032      |    100.2%    |
|       1.3        |    8    |   5.858 ± 0.023      |    100.2%    |
|       1.4        |    8    |   5.784 ± 0.035      |     98.9%    |

##### Atomistic system (palmitoyl tail only)

| `gorder` version | Threads | Analysis time [s]    | Rel. to v1.0 |
|:----------------:|:-------:|:--------------------:|:------------:|
|       1.0        |    1    |   10.615 ± 0.047     |    100.0%    |
|       1.1        |    1    |   10.761 ± 0.018     |    101.4%    |
|       1.2        |    1    |   10.738 ± 0.011     |    101.2%    |
|       1.3        |    1    |   10.749 ± 0.006     |    101.3%    |
|       1.4        |    1    |   10.465 ± 0.004     |     98.6%    |

---

#### Coarse-grained system

| `gorder` version | Threads | Analysis time [s]    | Rel. to v1.0 |
|:----------------:|:-------:|:--------------------:|:------------:|
|       1.0        |    1    |   4.745 ± 0.010      |    100.0%    |
|       1.1        |    1    |   4.736 ± 0.015      |     99.8%    |
|       1.2        |    1    |   4.720 ± 0.012      |     99.5%    |
|       1.3        |    1    |   4.719 ± 0.012      |     99.5%    |
|       1.4        |    1    |   4.450 ± 0.011      |     93.8%    |
|                                                                  |
|       1.0        |    8    |   1.986 ± 0.022      |    100.0%    |
|       1.1        |    8    |   1.894 ± 0.014      |     95.4%    |
|       1.2        |    8    |   1.893 ± 0.010      |     95.3%    |
|       1.3        |    8    |   1.966 ± 0.009      |     99.0%    |
|       1.4        |    8    |   1.912 ± 0.012      |     96.3%    |

---

#### United-atom system

| `gorder` version | Threads | Analysis time [s]    | Rel. to v1.0 |
|:----------------:|:-------:|:--------------------:|:------------:|
|       1.0        |    1    |   7.289 ± 0.087      |    100.0%    |
|       1.1        |    1    |   7.339 ± 0.021      |    100.7%    |
|       1.2        |    1    |   7.331 ± 0.011      |    100.6%    |
|       1.3        |    1    |   7.328 ± 0.014      |    100.5%    |
|       1.4        |    1    |   7.262 ± 0.012      |     99.6%    |
|                                                                  |
|       1.0        |    8    |   1.962 ± 0.017      |    100.0%    |
|       1.1        |    8    |   1.753 ± 0.018      |     89.3%    |
|       1.2        |    8    |   1.728 ± 0.011      |     88.1%    |
|       1.3        |    8    |   1.743 ± 0.016      |     88.8%    |
|       1.4        |    8    |   1.749 ± 0.016      |     89.1%    |

##### United-atom system (palmitoyl tail only)

| `gorder` version | Threads | Analysis time [s]    | Rel. to v1.0 |
|:----------------:|:-------:|:--------------------:|:------------:|
|       1.0        |    1    |   4.016 ± 0.015      |    100.0%    |
|       1.1        |    1    |   4.084 ± 0.017      |    101.7%    |
|       1.2        |    1    |   4.069 ± 0.008      |    101.3%    |
|       1.3        |    1    |   4.085 ± 0.011      |    101.7%    |
|       1.4        |    1    |   4.036 ± 0.005      |    100.5%    |