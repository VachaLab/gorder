## Benchmarking leaflet classification methods

Here are the benchmarking results for various leaflet classification methods using `gorder` v1.3. Classification was performed for every analyzed frame.

### Benchmarked trajectories
- **Atomistic (CHARMM36):** 256 POPC lipids, 64 500 atoms, 10 000 trajectory frames  
- **Coarse-grained (Martini 3):** 512 POPC lipids, 16 800 beads, 10 000 trajectory frames  

### System configuration
- **CPU:** 8-core Intel Core i7-11700  
- **SSD:** Samsung 870 EVO  
- **OS:**  GNU/Linux Mint 20.2  

### Compiler versions
- `rustc` 1.89.0, `gcc` 9.4.0

### Important analysis options
- Radius of the cylinder for local leaflet classification was 2.0 nm.

### Other information
- Benchmarked using `hyperfine` with cold cache.
- Analyses for 'No assignment', 'Global', 'Individual', 'SphericalClustering', and 'FromFile' were run 5 times each.
- Analyses for 'Local' and 'Clustering' were run once.

---

### Benchmarking results

#### Atomistic system

| Method          | Threads | Analysis time [s] | Rel. to No |
|:---------------:|:-------:|:-----------------:|:----------:|
| No assignment   |    1    | 16.207 ± 0.011    |   100%     |
| Global          |    1    | 28.103 ± 0.017    |   173%     |
| Local           |    1    | ~1130             |  7000%     |
| Individual      |    1    | 17.328 ± 0.014    |   107%     |
| SphClustering   |    1    | 19.296 ± 0.022    |   119%     |
| Clustering      |    1    | ~87               |   540%     |
| FromFile        |    1    | 17.765 ± 0.014    |   110%     |
|                                                            |
| No assignment   |    8    | 5.858 ± 0.023     |   100%     |
| Global          |    8    | 9.855 ± 0.032     |   168%     |
| Local           |    8    | ~508              |  8700%     |
| Individual      |    8    | 6.477 ± 0.018     |   111%     |
| SphClustering   |    8    | 6.690 ± 0.035     |   114%     |
| Clustering      |    8    | ~16               |   273%     |
| FromFile        |    8    | 6.897 ± 0.023     |   118%     |

---

#### Coarse-grained system

| Method         | Threads | Analysis time [s]  | Rel. to No |
|:--------------:|:-------:|:------------------:|:----------:|
| No assignment  |    1    | 4.719 ± 0.012      |   100%     |
| Global         |    1    | 7.090 ± 0.007      |   150%     |
| Local          |    1    | ~234               |  5000%     |
| Individual     |    1    | 4.970 ± 0.014      |   105%     |
| SphClustering  |    1    | 8.218 ± 0.011      |   174%     |
| FromFile       |    1    | 5.870 ± 0.007      |   124%     |
|                                                            |
| No assignment  |    8    | 1.986 ± 0.022      |   100%     |
| Global         |    8    | 2.213 ± 0.004      |   111%     | 
| Local          |    8    | ~39                |  2000%     |
| Individual     |    8    | 1.946 ± 0.003      |    98%     |
| SphClustering  |    8    | 2.254 ± 0.003      |   113%     |
| FromFile       |    8    | 2.889 ± 0.005      |   145%     |