// Released under MIT License.
// Copyright (c) 2024-2025 Ladislav Bartos

use gorder_core::input::Frequency as RsFreq;
use gorder_core::input::LeafletClassification as RsLeafletClassification;
use gorder_core::Leaflet;
use hashbrown::HashMap;
use numpy::ndarray::ArrayView2;
use numpy::PyArray2;
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::gen_stub_pyclass;
use pyo3_stub_gen::derive::gen_stub_pymethods;

use crate::string2axis;
use crate::Collect;
use crate::ConfigError;

macro_rules! try_extract {
    ($obj:expr, $( $t:ty ),*) => {
        $(
            if let Ok(classification_type) = $obj.extract::<$t>() {
                return Ok(Self(classification_type.0));
            }
        )*
    };
}

/// Helper structure for LeafletClassification.
#[derive(Clone)]
pub struct LeafletClassification(pub(crate) RsLeafletClassification);

impl<'source> FromPyObject<'source, '_> for LeafletClassification {
    type Error = PyErr;

    fn extract(obj: Borrowed<'source, '_, PyAny>) -> PyResult<Self> {
        try_extract!(
            obj,
            GlobalClassification,
            LocalClassification,
            IndividualClassification,
            ClusteringClassification,
            SphericalClusteringClassification,
            ManualClassification,
            NdxClassification
        );

        Err(ConfigError::new_err(
            "expected an instance of GlobalClassification, LocalClassification, IndividualClassification, ClusteringClassification, SphericalClusteringClassification, ManualClassification, or NdxClassification",
        ))
    }
}

/// Represents how often an action is performed.
///
/// Can specify that an action occurs once or at a regular interval (every N frames).
#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone)]
pub struct Frequency(RsFreq);

#[gen_stub_pymethods]
#[pymethods]
impl Frequency {
    /// Perform the action once.
    ///
    /// Returns
    /// -------
    /// Frequency
    ///     A frequency object representing a single execution.
    #[staticmethod]
    pub fn once() -> Self {
        Frequency(RsFreq::once())
    }

    /// Perform the action every N frames.
    ///
    /// Parameters
    /// ----------
    /// n_frames : int
    ///     Number of frames between each action.
    ///
    /// Returns
    /// -------
    /// Frequency
    ///     A frequency object representing the repeated execution interval.
    ///
    /// Raises
    /// ------
    /// ConfigError
    ///     If `n_frames` is 0.
    #[staticmethod]
    pub fn every(n_frames: usize) -> PyResult<Self> {
        Ok(Frequency(
            RsFreq::every(n_frames).map_err(|e| ConfigError::new_err(e.to_string()))?,
        ))
    }
}

/// Assign lipids into leaflets based on the global membrane center of geometry.
///
/// Reliable for planar membranes and fast. Recommended for most membranes.
///
/// Parameters
/// ----------
/// membrane : str
///     Selection query specifying all lipids forming the membrane.
/// heads : str
///     Selection query specifying reference atoms representing lipid headgroups
///     (typically phosphorus atoms or phosphate beads).
///     There must be exactly one such atom/bead per lipid molecule.
/// frequency : Optional[Frequency]
///     Frequency of classification. Defaults to every frame.
/// membrane_normal : Optional[str]
///     Membrane normal used for leaflet classification. Defaults to the membrane normal
///     specified for the entire Analysis.
/// collect : Optional[Union[bool, str]], default=False
///     Determines whether leaflet classification data are saved and exported.
///     By default (`False`), data are not saved.
///     If `True`, data are saved internally and accessible via the Python API, but not written to a file.
///     If a string is provided, data are saved and written to the specified output file.
/// flip : bool, default=False
///     Flip the assignment. Upper leaflet becomes lower leaflet and vice versa. The default value is False.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.leaflets")]
#[derive(Clone)]
pub struct GlobalClassification(RsLeafletClassification);

#[gen_stub_pymethods]
#[pymethods]
impl GlobalClassification {
    #[new]
    #[pyo3(signature = (membrane, heads, frequency = None, membrane_normal = None, collect = None, flip = false))]
    pub fn new<'a>(
        membrane: &str,
        heads: &str,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[gorder.Frequency]", imports=("typing")
        ))]
        frequency: Option<Frequency>,
        membrane_normal: Option<&str>,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[typing.Union[builtins.bool, builtins.str]]", imports=("typing")
        ))]
        collect: Option<Bound<'a, PyAny>>,
        flip: bool,
    ) -> PyResult<Self> {
        let classification = add_collect(
            add_normal(
                add_freq(RsLeafletClassification::global(membrane, heads), frequency)?,
                membrane_normal,
            )?,
            collect,
        )?;

        Ok(Self(classification.with_flip(flip)))
    }
}

/// Assign lipids into leaflets based on the local membrane center of geometry.
///
/// Reliable for planar membranes but slow.
/// Recommended for planar membranes if the global and individual methods do not work for you.
///
/// Parameters
/// ----------
/// membrane : str
///     Selection query specifying all lipids forming the membrane.
/// heads : str
///     Selection query specifying reference atoms representing lipid headgroups
///     (typically phosphorus atoms or phosphate beads).
///     There must be exactly one such atom/bead per lipid molecule.
/// radius : float
///     Radius of a cylinder used to calculate the local membrane center of geometry (in nm).
/// frequency : Optional[Frequency]
///     Frequency of classification. Defaults to every frame.
/// membrane_normal : Optional[str]
///     Membrane normal used for leaflet classification. Defaults to the membrane normal
///     specified for the entire Analysis.
/// collect : Optional[Union[bool, str]], default=False
///     Determines whether leaflet classification data are saved and exported.
///     By default (`False`), data are not saved.
///     If `True`, data are saved internally and accessible via the Python API, but not written to a file.
///     If a string is provided, data are saved and written to the specified output file.
/// flip : bool, default=False
///     Flip the assignment. Upper leaflet becomes lower leaflet and vice versa. The default value is False.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.leaflets")]
#[derive(Clone)]
pub struct LocalClassification(RsLeafletClassification);

#[gen_stub_pymethods]
#[pymethods]
impl LocalClassification {
    #[new]
    #[pyo3(signature = (membrane, heads, radius, frequency = None, membrane_normal = None, collect = None, flip = false))]
    pub fn new<'a>(
        membrane: &str,
        heads: &str,
        radius: f32,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[gorder.Frequency]", imports=("typing")
        ))]
        frequency: Option<Frequency>,
        membrane_normal: Option<&str>,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[typing.Union[builtins.bool, builtins.str]]", imports=("typing")
        ))]
        collect: Option<Bound<'a, PyAny>>,
        flip: bool,
    ) -> PyResult<Self> {
        if radius <= 0.0 {
            return Err(ConfigError::new_err(format!(
                "radius must be greater than 0, not `{}`.",
                radius
            )));
        }

        let classification = add_collect(
            add_normal(
                add_freq(
                    RsLeafletClassification::local(membrane, heads, radius),
                    frequency,
                )?,
                membrane_normal,
            )?,
            collect,
        )?;

        Ok(Self(classification.with_flip(flip)))
    }
}

/// Assign lipids into leaflets based on the orientation of acyl chains.
///
/// Less reliable but very fast. Recommended for very large, undulating planar membranes.
///
/// Parameters
/// ----------
/// heads : str
///     Selection query specifying reference atoms representing lipid headgroups
///     (typically phosphorus atoms or phosphate beads).
///     There must be exactly one such atom/bead per lipid molecule.
/// methyls : str
///     Selection query specifying reference atoms representing methyl groups at the ends of lipid tails.
///     There should be exactly one such atom/bead per acyl chain (e.g., two for lipids with two acyl chains).
/// frequency : Optional[Frequency]
///     Frequency of classification. Defaults to every frame.
/// membrane_normal : Optional[str]
///     Membrane normal used for leaflet classification. Defaults to the membrane normal
///     specified for the entire Analysis.
/// collect : Optional[Union[bool, str]], default=False
///     Determines whether leaflet classification data are saved and exported.
///     By default (`False`), data are not saved.
///     If `True`, data are saved internally and accessible via the Python API, but not written to a file.
///     If a string is provided, data are saved and written to the specified output file.
/// flip : bool, default=False
///     Flip the assignment. Upper leaflet becomes lower leaflet and vice versa. The default value is False.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.leaflets")]
#[derive(Clone)]
pub struct IndividualClassification(RsLeafletClassification);

#[gen_stub_pymethods]
#[pymethods]
impl IndividualClassification {
    #[new]
    #[pyo3(signature = (heads, methyls, frequency = None, membrane_normal = None, collect = None, flip = false))]
    pub fn new<'a>(
        heads: &str,
        methyls: &str,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[gorder.Frequency]", imports=("typing")
        ))]
        frequency: Option<Frequency>,
        membrane_normal: Option<&str>,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[typing.Union[builtins.bool, builtins.str]]", imports=("typing")
        ))]
        collect: Option<Bound<'a, PyAny>>,
        flip: bool,
    ) -> PyResult<Self> {
        let classification = add_collect(
            add_normal(
                add_freq(
                    RsLeafletClassification::individual(heads, methyls),
                    frequency,
                )?,
                membrane_normal,
            )?,
            collect,
        )?;

        Ok(Self(classification.with_flip(flip)))
    }
}

/// Assign lipids into leaflets using spectral clustering.
///
/// Reliable but very slow. Recommended for curved membranes, tubes and vesicles.
///
/// Parameters
/// ----------
/// heads : str
///     Selection query specifying reference atoms representing lipid headgroups
///     (typically phosphorus atoms or phosphate beads).
///     There must be exactly one such atom/bead per lipid molecule.
/// frequency : Optional[Frequency]
///     Frequency of classification. Defaults to every frame.
/// collect : Optional[Union[bool, str]], default=False
///     Determines whether leaflet classification data are saved and exported.
///     By default (`False`), data are not saved.
///     If `True`, data are saved internally and accessible via the Python API, but not written to a file.
///     If a string is provided, data are saved and written to the specified output file.
/// flip : bool, default=False
///     Flip the assignment. Upper leaflet becomes lower leaflet and vice versa. The default value is False.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.leaflets")]
#[derive(Clone)]
pub struct ClusteringClassification(RsLeafletClassification);

#[gen_stub_pymethods]
#[pymethods]
impl ClusteringClassification {
    #[new]
    #[pyo3(signature = (heads, frequency = None, collect = None, flip = false))]
    pub fn new<'a>(
        heads: &str,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[gorder.Frequency]", imports=("typing")
        ))]
        frequency: Option<Frequency>,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[typing.Union[builtins.bool, builtins.str]]", imports=("typing")
        ))]
        collect: Option<Bound<'a, PyAny>>,
        flip: bool,
    ) -> PyResult<Self> {
        let classification = add_collect(
            add_freq(RsLeafletClassification::clustering(heads), frequency)?,
            collect,
        )?;

        Ok(Self(classification.with_flip(flip)))
    }
}

/// Assign lipids into leaflets using gaussian mixture model.
///
/// Reliable for spherical vesicles and fast. Do not use for anything other than vesicles!
///
/// Parameters
/// ----------
/// heads : str
///     Selection query specifying reference atoms representing lipid headgroups
///     (typically phosphorus atoms or phosphate beads).
///     There must be exactly one such atom/bead per lipid molecule.
/// frequency : Optional[Frequency]
///     Frequency of classification. Defaults to every frame.
/// collect : Optional[Union[bool, str]], default=False
///     Determines whether leaflet classification data are saved and exported.
///     By default (`False`), data are not saved.
///     If `True`, data are saved internally and accessible via the Python API, but not written to a file.
///     If a string is provided, data are saved and written to the specified output file.
/// flip : bool, default=False
///     Flip the assignment. Upper leaflet becomes lower leaflet and vice versa. The default value is False.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.leaflets")]
#[derive(Clone)]
pub struct SphericalClusteringClassification(RsLeafletClassification);

#[gen_stub_pymethods]
#[pymethods]
impl SphericalClusteringClassification {
    #[new]
    #[pyo3(signature = (heads, frequency = None, collect = None, flip = false))]
    pub fn new<'a>(
        heads: &str,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[gorder.Frequency]", imports=("typing")
        ))]
        frequency: Option<Frequency>,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[typing.Union[builtins.bool, builtins.str]]", imports=("typing")
        ))]
        collect: Option<Bound<'a, PyAny>>,
        flip: bool,
    ) -> PyResult<Self> {
        let classification = add_collect(
            add_freq(
                RsLeafletClassification::spherical_clustering(heads),
                frequency,
            )?,
            collect,
        )?;

        Ok(Self(classification.with_flip(flip)))
    }
}

/// Get leaflet assignment from an external YAML file or a dictionary.
///
/// Parameters
/// ----------
/// input : Union[str, Mapping[str, ndarray[uint8]]]
///     Path to the input YAML file containing the leaflet assignment
///     or a dictionary specifying the leaflet assignment.
/// frequency : Optional[Frequency]
///     Frequency of classification. Defaults to every frame.
/// flip : bool, default=False
///     Flip the assignment. Upper leaflet becomes lower leaflet and vice versa. The default value is False.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.leaflets")]
#[derive(Clone)]
pub struct ManualClassification(RsLeafletClassification);

#[gen_stub_pymethods]
#[pymethods]
impl ManualClassification {
    #[new]
    #[pyo3(signature = (input, frequency = None, flip = false))]
    pub fn new(
        #[gen_stub(override_type(
            type_repr = "typing.Union[builtins.str, typing.Mapping[builtins.str, numpy.typing.NDArray[numpy.uint8]]]", imports=("typing", "numpy")
        ))]
        input: &Bound<'_, PyAny>,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[gorder.Frequency]", imports=("typing")
        ))]
        frequency: Option<Frequency>,
        flip: bool,
    ) -> PyResult<Self> {
        let classification = if let Ok(file) = input.extract::<String>() {
            RsLeafletClassification::from_file(&file)
        } else if let Ok(map) = extract_map(input) {
            let converted_map = convert_leaflet_map(map)?;
            RsLeafletClassification::from_map(converted_map)
        } else {
            return Err(ConfigError::new_err(
                "invalid type for ManualClassification input: expected str or Mapping",
            ));
        };

        Ok(Self(add_freq(classification, frequency)?.with_flip(flip)))
    }
}

/// Get leaflet assignment from NDX file(s).
///
/// Parameters
/// ----------
/// ndx : Sequence[str]
///     A list of NDX files to read.
/// heads : str
///     Selection query specifying reference atoms representing lipid headgroups
///     (typically phosphorus atoms or phosphate beads).
///     There must be exactly one such atom/bead per lipid molecule.
/// upper_leaflet : str
///     Name of the group in the NDX file(s) specifying the atoms of the upper leaflet.
/// lower_leaflet : str
///     Name of the group in the NDX file(s) specifying the atoms of the lower leaflet.
/// frequency : Optional[Frequency]
///     Frequency of classification. Defaults to every frame.
/// flip : bool, default=False
///     Flip the assignment. Upper leaflet becomes lower leaflet and vice versa. The default value is False.
///
/// Notes
/// -----
/// - No glob expansion is performed for the NDX files.
#[gen_stub_pyclass]
#[pyclass(module = "gorder.leaflets")]
#[derive(Clone)]
pub struct NdxClassification(RsLeafletClassification);

#[gen_stub_pymethods]
#[pymethods]
impl NdxClassification {
    #[new]
    #[pyo3(signature = (ndx, heads, upper_leaflet, lower_leaflet, frequency = None, flip = false))]
    pub fn new(
        ndx: Vec<String>,
        heads: &str,
        upper_leaflet: &str,
        lower_leaflet: &str,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[gorder.Frequency]", imports=("typing")
        ))]
        frequency: Option<Frequency>,
        flip: bool,
    ) -> PyResult<Self> {
        let classification = add_freq(
            RsLeafletClassification::from_ndx(
                &ndx.iter().map(String::as_str).collect::<Vec<&str>>(),
                heads,
                upper_leaflet,
                lower_leaflet,
            ),
            frequency,
        )?;

        Ok(Self(classification.with_flip(flip)))
    }
}

/// Attempt to add frequency to leaflet classification.
fn add_freq(
    mut classification: RsLeafletClassification,
    frequency: Option<Frequency>,
) -> PyResult<RsLeafletClassification> {
    if let Some(freq) = frequency {
        classification = classification.with_frequency(freq.0);
    }

    Ok(classification)
}

/// Attempt to add membrane normal to leaflet classification.
fn add_normal(
    mut classification: RsLeafletClassification,
    normal: Option<&str>,
) -> PyResult<RsLeafletClassification> {
    if let Some(normal) = normal {
        classification = classification.with_membrane_normal(string2axis(normal)?);
    }

    Ok(classification)
}

/// Attempt to add request for data collection to leaflet classification.
fn add_collect<'a>(
    classification: RsLeafletClassification,
    collect: Option<Bound<'a, PyAny>>,
) -> PyResult<RsLeafletClassification> {
    if let Some(collect) = collect {
        match classification {
            RsLeafletClassification::Global(_) |
            RsLeafletClassification::Local(_) |
            RsLeafletClassification::Individual(_) |
            RsLeafletClassification::Clustering(_) |
            RsLeafletClassification::SphericalClustering(_) => return Ok(classification.with_collect(Collect::extract(collect.as_borrowed())?.0)),
            RsLeafletClassification::FromFile(_) |
            RsLeafletClassification::FromMap(_) |
            RsLeafletClassification::FromNdx(_) => return Err(ConfigError::new_err("collecting leaflet classification data for manual leaflet classification methods is not supported"))
        }
    }

    Ok(classification)
}

/// Converts a Python dictionary whose keys are strings and values are 2D numpy arrays
/// into a hashbrown::HashMap<String, Vec<Vec<u8>>>.
fn extract_map(py_obj: &Bound<'_, PyAny>) -> PyResult<HashMap<String, Vec<Vec<u8>>>> {
    let dict = py_obj.cast::<PyDict>().map_err(|_| {
        ConfigError::new_err(
            "expected a dictionary using molecule types as keys and 2D numpy arrays with shape (n_frames, n_molecules) as values",
        )
    })?;
    let mut map = HashMap::new();
    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;
        let nested_vec = extract_nested_vector(&value)?;
        map.insert(key_str, nested_vec);
    }
    Ok(map)
}

/// Converts a two-dimensional numpy array into a Vec<Vec<u8>>.
fn extract_nested_vector(py_obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<u8>>> {
    let array = py_obj
        .cast::<PyArray2<u8>>()
        .map_err(|_| ConfigError::new_err("expected a 2D numpy array for leaflet assignment"))?;

    let array_view: ArrayView2<u8> = unsafe { array.as_array() };

    let vec: Vec<Vec<u8>> = array_view.outer_iter().map(|row| row.to_vec()).collect();

    Ok(vec)
}

/// Convert `HashMap<String, Vec<Vec<u8>>>` to `HashMap<String, Vec<Vec<Leaflet>>>`.
/// 1 => Leaflet::Upper.
/// 0 => Leaflet::Lower.
fn convert_leaflet_map(
    input: HashMap<String, Vec<Vec<u8>>>,
) -> PyResult<HashMap<String, Vec<Vec<Leaflet>>>> {
    input
        .into_iter()
        .map(|(key, matrix)| {
            let converted_matrix: PyResult<Vec<Vec<Leaflet>>> = matrix
                .into_iter()
                .map(|row| row.into_iter().map(|number| match number {
                    1 => Ok(Leaflet::Upper),
                    0 => Ok(Leaflet::Lower),
                    x => Err(ConfigError::new_err(
                        format!("'{}' is not a valid leaflet identifier (use 0 for lower leaflet, and 1 for upper leaflet)", x)))
                }).collect())
                .collect();

            converted_matrix.map(|m| (key, m))
        })
        .collect()
}
