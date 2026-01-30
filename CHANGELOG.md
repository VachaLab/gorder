## Version 1.4.1
### Bug fixes
- Fixed a bug where the `collect`/`export` keyword would not be parsed correctly if its value was `false`.

## Version 1.4.0
### Inverting geometric selections
- Geometry selections can be now inverted using the `invert` keyword. See [the manual](https://vachalab.github.io/gorder-manual/geometry.html#inverting-the-selection) for more details.

### Other changes
- A companion tool [GUIorder](https://github.com/VachaLab/guiorder) providing graphical user interface for `gorder` has been released. See [the manual](https://vachalab.github.io/gorder-manual/guiorder.html) for more details.
- `gorder` along with its documentation has been transferred to [Vacha Lab organization](https://github.com/VachaLab).

## Version 1.3.0
### Faster leaflet assignment for vesicles
- Implemented a new leaflet classification method, `SphericalClustering`, exclusively for vesicles. This method classifies lipids using a 2-component Gaussian Mixture Model based on the distances between lipid headgroups and the vesicle center of geometry. This method is **much** faster than `Clustering`, but is only usable for systems with a single unilamellar vesicle. See [the manual](https://vachalab.github.io/gorder-manual/leaflets.html#spherical-clustering-method) for more details.

### Analyzing very long trajectories
- Trajectories longer than 2 147 483 647 simulation steps can be now analyzed.

## Version 1.2.0
### Exporting internal data
- You can now collect and export information about which leaflets individual lipids were assigned to during the analysis, as well as dynamically calculated membrane normals, to external files. This is only performed when requested in the analysis setup. The data can also be accessed via the Rust and Python APIs. Read more in the corresponding sections of the manual: [Exporting lipid assignment data](https://vachalab.github.io/gorder-manual/leaflets_export.html) and [Exporting membrane normals](https://vachalab.github.io/gorder-manual/normals_export.html).

### Flipping the assignment
- When assigning lipids to leaflets, you can use the `flip` keyword, which reverses the assignment—upper leaflet becomes lower leaflet and vice versa. This may seem useless, but it can help with leaflet assignment when using the clustering method. See [the manual](https://vachalab.github.io/gorder-manual/leaflets.html#flipping-the-assignment) for more details.

### Python API
- The Python `gorder` library now exposes type information and supports autocomplete.
- [Documentation of the Python API](https://vachalab.github.io/pygorder-docs/) has been improved and should be much more complete.

### Bug fixes
- Manually assigning membrane normals previously returned an error unless the analysis step was set to 1. This has been fixed, and manual membrane normals can now be used with any analysis step.

### Other changes
- Changed the wording of some error messages and adjusted the formatting of certain output written to stdout.

## Version 1.1.0
- The `@membrane` macro now matches a wider variety of lipid types. See the [GSL guide](https://vachalab.github.io/gsl-guide/autodetection.html) for details on how the macro currently expands.
  - This change makes selecting lipid molecules that form your membrane simpler and more reliable, especially when working with less common lipid types.
  - The macro still cannot guarantee identification of all lipid molecules, so exercise caution when using it.
  - **Since the `@membrane` macro now matches more lipid types, some queries using this macro may return different atom selections. Therefore, `gorder` v1.1.0 is not guaranteed to return the same results as `gorder` v1.0.0 or previous versions.**
  - Example of behavior change and required modification: In the [gorder article](https://doi.org/10.1016/j.softx.2025.102254) (Figure 3), we used the query `(@membrane and name r'C3.+|C2.+') or (resname PVCL2 and name r'C[ABCD].+')` to select all lipid carbons in a complex system. In previous versions of `gorder`, PVCL2 was not identified as a membrane lipid when using the `@membrane` macro. This behavior has now changed, and the PVCL2 molecule **is matched** by the `@membrane` macro. In `gorder` v1.1.0, the above query should therefore be modified to `(@membrane and not resname PVCL2 and name r'C3.+|C2.+') or (resname PVCL and name r'C[ABCD].+')` (this is one of the rare cases where the update complicates things, as we don't want the C3 and C2 atoms of PVCL2 to be selected).

## Version 1.0.0
- Support for GROMACS 2025 TPR files.
- More precise calculation of geometric center of groups using the Refined Bai-Breen algorithm. **Note that as a consequence of this, `gorder 1.0.0` may return slightly different results for some systems than previous versions.**
- Providing the current working directory as the directory for saving ordermaps will now result in an error.
- Changes to the Rust API: Made all user-provided parameters of all methods for leaflet assignment public.

***

## Version 0.7.1
- Fixed undefined behavior in the `groan_rs` library's C source code that caused TRR files to be read incorrectly when compiled with certain C compilers (e.g., clang).

## Version 0.7.0
- **Leaflet classification for curved membranes:** `gorder` can now use spectral clustering to classify lipids into leaflets for any membrane geometry, including buckled membranes and vesicles. See the [manual](https://vachalab.github.io/gorder-manual/leaflets.html#clustering-method-for-leaflet-classification) for more information.
- **!!BREAKING CHANGE!!** **Removed support for some trajectory formats:** Removed support for PDB, Amber NetCDF, DCD, and LAMMPSTRJ trajectories. (PDB structure files are still supported.) The parsing of these trajectories relied on the `chemfiles` library, whose maintenance standards and capabilities just do not meet the quality requirements for `gorder`. If you need to use these trajectory formats, you can still use `gorder v0.6`.

***

## Version 0.6.0
- **United-atom order parameters:** `gorder` is now able to calculate order parameters in united-atom systems. See the [manual](https://vachalab.github.io/gorder-manual/uaorder_basics.html) for more information.
- **Python API:** `gorder` is now available as a Python package. See the [manual](https://vachalab.github.io/gorder-manual/python_api.html) for more information.
- **Trajectory concatenation:** You can now provide multiple trajectory files which will be all joined into one trajectory and analyzed. In case there are duplicate frames at trajectory boundaries, `gorder` will analyze only one of the duplicate frames. This feature is currently only supported for XTC and TRR files. See the [manual](https://vachalab.github.io/gorder-manual/multiple_trajectories.html) for more information.
- **Manual membrane normals:** Membrane normals can be now assigned manually for each lipid molecule in each trajectory frame. See the [manual](https://vachalab.github.io/gorder-manual/manual_normals.html) for more information.
- **Optimizations:** Local leaflet classification method and dynamic local membrane normal calculation are now much faster, especially for very large systems, through using cell lists. **Warning:** The changes may cause small differences in the calculated order parameters compared to version 0.5, especially for very short trajectories. These differences should be on the order of 0.0001 arb. u. or lower.

***

## Version 0.5.0
- **Dynamic membrane normal calculation:** Membrane normals can be now calculated dynamically from actual membrane shape which allows the calculation of order parameters for vesicles and similar systems (see the [manual](https://vachalab.github.io/gorder-manual/membrane_normal.html)).
- **Ignoring PBC:** You can now choose to ignore periodic boundary conditions. This allows analyzing simulations with non-orthogonal simulation boxes with some small additional friction (making molecules whole). See the [manual](https://vachalab.github.io/gorder-manual/no_pbc.html) for more information.
- **Reworked manual lipid assignment:** It is now possible to classify lipids into membrane leaflets using NDX files, enabling integration of `gorder` with `FATSLiM`. **BREAKING CHANGE** The keywords to request manual leaflet assignment using a leaflet assignment file have been changed, see the [manual](https://vachalab.github.io/gorder-manual/manual_leaflets.html#assigning-lipids-using-a-leaflet-assignment-file) for more information.
- **Ordermaps visualization:** A python script is now generated inside any created `ordermaps` directory which can be used to easily plot the ordermaps. Changed the default range of the colorbar in ordermaps to more reasonable values.
- **More trajectory formats:** Added **experimental** support for more trajectory formats, namely TRR, GRO, PDB, Amber NetCDF, DCD, and LAMMPSTRJ. Always prefer using XTC trajectory as `gorder` is optimized to read it very fast. There are also some limitations connected with using different trajectory formats, see the [manual](https://vachalab.github.io/gorder-manual/other_input.html#trajectory-file-formats).
- **Bug fixes and other changes:**
  - `gorder` now returns an error if the center of geometry calculation for leaflet classification is nonsensical (i.e., `nan`).
  - If molecule classification runs longer than expected, progress is logged. By default, progress output begins after 500 ms, but you can adjust this delay using the environment variable `GORDER_MOLECULE_CLASSIFICATION_TIME_LIMIT`.
  - Added more color to information written to standard output during analysis. Changed logging for output file writing. Changed logging for molecule types.
  - Reading an ndx file with invalid or duplicated group names no longer results in a hard error but instead raises a warning.

***

## Version 0.4.0
- **Geometry selection:** Added the ability to select a geometric region for analysis. Users can now specify cuboidal, spherical, or cylindrical regions, and order parameters will be calculated only for bonds located within the selected region.
- **Support for reading GRO, PDB, and PQR files:** These file formats are now supported as input structure files. In some cases, an additional "bonds" file specifying the system's connectivity may be required. Refer to the manual for more details.
- **Manual assignment of lipids to leaflets:** Lipids can now be manually assigned to leaflets using a provided leaflet assignment file. Refer to the manual for detailed instructions.
- **Calculating average results for the entire system:** YAML and TAB files now include information about the average order parameters calculated across all bonds and molecule types in the system. Additionally, ordermaps are generated for the entire system.

***

## Version 0.3.0
- **Error estimation and convergence analysis**: Implemented error estimation and convergence analysis. Refer to the corresponding section of the manual for more details.
- **Leaflet classification**: Leaflet classification can now also be performed either every N analyzed trajectory frames or only once at the start of the analysis.
- **Improved XTC file reading**: Switched to using the `molly` crate for reading XTC files. This allows reading only the parts of the XTC files that are needed, making `gorder` more than twice as fast compared to version 0.2.
- **Reworked Rust API**: The Rust API has been restructured, enabling access to results without the need to write output files.
- **Enhanced output files**:
  - YAML and TAB files now display the average order parameter calculated from all bonds of a single molecule type.
  - An average ordermap, summarizing order parameters collected from all bonds of a single molecule type, is now automatically generated during ordermap analysis.
- **YAML format updates**: The YAML format has been revised: molecule types are no longer stored as a list but are instead represented as a dictionary.
- **Export configuration**: Analysis parameters can now be exported into a YAML file using the `--export-config` argument.
- **Updated heavy atom order parameter calculation**: Adjusted the calculation of average order parameters for heavy atoms. This may result in order parameters being available for atoms even if all bond order parameters are reported as NaN. (And that's okay.)
- **Bug fixes**:
  - Fixed a minor issue causing occasional rounding discrepancies between YAML and other output formats, which could lead to slightly different results.
  - Corrected CG bond numbering in XVG files, ensuring they are properly numbered starting from 1.
- **Improved error messages**: Certain error messages have been clarified to enhance their readability and comprehension.
