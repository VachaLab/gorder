// Released under MIT License.
// Copyright (c) 2024-2025 Ladislav Bartos

//! Contains structures and methods for the assignment of lipids into membrane leaflets.

use std::fmt::{self, Write};

use colored::Colorize;
use getset::{CopyGetters, Getters};
use hashbrown::HashMap;
use serde::{
    de::{self, MapAccess, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
use serde_yaml::Value;

use crate::{input::Collect, Leaflet};

use super::{frequency::Frequency, Axis};

/// Parameters for the classification of lipids into membrane leaflets.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub enum LeafletClassification {
    Global(GlobalParams),
    Local(LocalParams),
    Individual(IndividualParams),
    FromFile(FromFileParams),
    #[serde(alias = "Inline")]
    FromMap(FromMapParams),
    FromNdx(FromNdxParams),
    Clustering(ClusteringParams),
    SphericalClustering(SphericalClusteringParams),
}

impl fmt::Display for LeafletClassification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LeafletClassification::Global(_) => write!(f, "global"),
            LeafletClassification::Local(_) => write!(f, "local"),
            LeafletClassification::Individual(_) => write!(f, "individual"),
            LeafletClassification::FromFile(_)
            | LeafletClassification::FromMap(_)
            | LeafletClassification::FromNdx(_) => write!(f, "manual"),
            LeafletClassification::Clustering(_) => write!(f, "clustering"),
            LeafletClassification::SphericalClustering(_) => write!(f, "spherical clustering"),
        }
    }
}

impl LeafletClassification {
    /// Classify lipids based on the global membrane center of geometry.
    /// Generally reliable and fast. The best option when working with disrupted membranes.
    ///
    /// ## Parameters
    /// - `membrane` - selection of all lipids forming the membrane
    /// - `heads` - reference atoms identifying lipid headgroups (usually a phosphorus atom or a phosphate bead);
    ///    there must only be one such atom/bead per lipid molecule.
    pub fn global(membrane: &str, heads: &str) -> LeafletClassification {
        Self::Global(GlobalParams {
            membrane: membrane.to_string(),
            heads: heads.to_string(),
            frequency: Frequency::default(),
            membrane_normal: None,
            collect: Collect::default(),
            flip: Default::default(),
        })
    }

    /// Classify lipids based on the local membrane center of geometry.
    /// Useful for curved membranes, very slow.
    ///
    /// ## Parameters
    /// - `membrane` - selection of all lipids forming the membrane
    /// - `heads` - reference atoms identifying lipid headgroups (usually a phosphorus atom or a phosphate bead);
    ///    there must only be one such atom/bead per lipid molecule.
    /// - `radius` - radius of a cylinder for the calculation of local membrane center of geometry (in nm)
    ///
    /// ## Panic
    /// Panics if the radius is not positive.
    pub fn local(membrane: &str, heads: &str, radius: f32) -> LeafletClassification {
        if radius <= 0.0 {
            panic!("Radius must be greater than 0, not `{}`.", radius);
        }

        Self::Local(LocalParams {
            membrane: membrane.to_string(),
            heads: heads.to_string(),
            radius,
            frequency: Frequency::default(),
            membrane_normal: None,
            collect: Collect::default(),
            flip: Default::default(),
        })
    }

    /// Classify lipids based on the orientation of acyl chains.
    /// Less reliable but fast.
    ///
    /// ## Parameters
    /// - `heads`: reference atoms identifying lipid headgroups (usually a phosphorus atom or a phosphate bead);
    ///    there must only be one such atom/bead per lipid molecule.
    /// - `methyls`: reference atoms identifying methyl groups of lipid tails, i.e., the ends of lipid tails;
    ///    there should be only one such atom/bead per one acyl chain in the molecule (e.g., two for lipids with two acyl chains).
    #[inline(always)]
    pub fn individual(heads: &str, methyls: &str) -> LeafletClassification {
        Self::Individual(IndividualParams {
            heads: heads.to_string(),
            methyls: methyls.to_string(),
            frequency: Frequency::default(),
            membrane_normal: None,
            collect: Collect::default(),
            flip: Default::default(),
        })
    }

    /// Read leaflet assignment from an external yaml file.
    ///
    /// ## Parameters
    /// - `file`: path to the input yaml file containing the leaflet assignment.
    #[inline(always)]
    pub fn from_file(file: &str) -> LeafletClassification {
        Self::FromFile(FromFileParams {
            file: file.to_owned(),
            frequency: Frequency::default(),
            flip: Default::default(),
        })
    }

    /// Provide leaflet assignment as a hash map.
    #[inline(always)]
    pub fn from_map(assignment: HashMap<String, Vec<Vec<Leaflet>>>) -> LeafletClassification {
        Self::FromMap(FromMapParams {
            assignment,
            frequency: Frequency::default(),
            flip: Default::default(),
        })
    }

    /// Get leaflet assignment using NDX file(s).
    ///
    /// ## Parameters
    /// - `ndx`: a vector of NDX files to read
    /// - `heads`: GSL query specifying the atoms to use as head identifiers of molecules
    /// - `upper_leaflet`: name of the group in the NDX file(s) specifying the atoms of the upper leaflet
    /// - `lower_leaflet`: name of the group in the NDX file(s) specifying the atoms of the lower leaflet
    ///
    /// ## Notes
    /// - No glob expansion is performed for the NDX files.
    #[inline(always)]
    pub fn from_ndx(
        ndx: &[&str],
        heads: &str,
        upper_leaflet: &str,
        lower_leaflet: &str,
    ) -> LeafletClassification {
        Self::FromNdx(FromNdxParams {
            ndx: ndx.iter().map(|x| (*x).to_owned()).collect(),
            heads: heads.to_owned(),
            upper_leaflet: upper_leaflet.to_owned(),
            lower_leaflet: lower_leaflet.to_owned(),
            frequency: Frequency::default(),
            flip: Default::default(),
        })
    }

    /// Get leaflet assignment using spectral clustering.
    ///
    /// This is the only automatic method applicable to
    /// membranes with any curved geometry
    /// (vesicles, buckled membranes, tubes).
    ///
    /// ## Parameters
    /// - `heads`: GSL query specifying the atoms to use as head identifiers of molecules
    pub fn clustering(heads: &str) -> LeafletClassification {
        Self::Clustering(ClusteringParams {
            heads: heads.to_owned(),
            frequency: Frequency::default(),
            collect: Collect::default(),
            flip: Default::default(),
        })
    }

    /// Get leaflet assignment using gaussian mixture model.
    ///
    /// This method is ONLY applicable to unilamellar vesicles.
    ///
    /// ## Parameters
    /// - `heads`: GSL query specifying the atoms to use as head identifiers of molecules
    pub fn spherical_clustering(heads: &str) -> LeafletClassification {
        Self::SphericalClustering(SphericalClusteringParams {
            heads: heads.to_owned(),
            frequency: Frequency::default(),
            collect: Collect::default(),
            flip: Default::default(),
        })
    }

    /// Assign lipids to leaflets every N analyzed trajectory frames or only once (using the first trajectory frame).
    /// (Note that this is 'analyzed trajectory frames' - if you skip some frames using `step`,
    /// they will not be counted here.)
    #[inline(always)]
    pub fn with_frequency(mut self, frequency: Frequency) -> Self {
        match &mut self {
            LeafletClassification::Global(x) => x.frequency = frequency,
            LeafletClassification::Local(x) => x.frequency = frequency,
            LeafletClassification::Individual(x) => x.frequency = frequency,
            LeafletClassification::FromFile(x) => x.frequency = frequency,
            LeafletClassification::FromMap(x) => x.frequency = frequency,
            LeafletClassification::FromNdx(x) => x.frequency = frequency,
            LeafletClassification::Clustering(x) => x.frequency = frequency,
            LeafletClassification::SphericalClustering(x) => x.frequency = frequency,
        }

        self
    }

    /// Set the membrane normal for leaflet classification.
    /// If not set, the globally specified membrane normal (Z-axis by default) is used.
    /// You only need to set this value when using a dynamic membrane normal as the global one.
    ///
    /// The method has no effect if you are assigning lipids manually into leaflets or if you are using clustering.
    #[inline(always)]
    pub fn with_membrane_normal(mut self, membrane_normal: Axis) -> Self {
        match &mut self {
            LeafletClassification::Global(x) => x.membrane_normal = Some(membrane_normal),
            LeafletClassification::Local(x) => x.membrane_normal = Some(membrane_normal),
            LeafletClassification::Individual(x) => x.membrane_normal = Some(membrane_normal),
            // ignore for manual classification
            LeafletClassification::FromFile(_)
            | LeafletClassification::FromMap(_)
            | LeafletClassification::FromNdx(_) => (),
            // ignore for clustering
            LeafletClassification::Clustering(_) => (),
            LeafletClassification::SphericalClustering(_) => (),
        }

        self
    }

    /// Collect and store leaflet classification data.
    /// If `true`, the data are collected but only accessible using API.
    /// If a string is provided, the data will be exported into the output file.
    ///
    /// ## Panics
    /// This method is not supported for manual leaflet classification methods
    /// (FromFile, FromMap, FromNdx) and will panic if called.
    #[inline(always)]
    pub fn with_collect(mut self, collect: impl Into<Collect>) -> Self {
        match &mut self {
            LeafletClassification::Global(x) => x.collect = collect.into(),
            LeafletClassification::Local(x) => x.collect = collect.into(),
            LeafletClassification::Individual(x) => x.collect = collect.into(),
            LeafletClassification::Clustering(x) => x.collect = collect.into(),
            LeafletClassification::SphericalClustering(x) => x.collect = collect.into(),
            // panic for manual classification
            LeafletClassification::FromFile(_) | LeafletClassification::FromMap(_) | LeafletClassification::FromNdx(_) =>
                panic!("Collecting leaflet classification data is not supported for manual leaflet classification.\nThe data are already collected!"),
        }

        self
    }

    /// Treat the `upper` leaflet as the `lower` leaflet and vice versa.
    /// Typically only useful for clustering classification if the leaflets get labeled
    /// differently than you want.
    pub fn with_flip(mut self, flip: bool) -> Self {
        match &mut self {
            LeafletClassification::Global(x) => x.flip = flip,
            LeafletClassification::Local(x) => x.flip = flip,
            LeafletClassification::Individual(x) => x.flip = flip,
            LeafletClassification::FromFile(x) => x.flip = flip,
            LeafletClassification::FromMap(x) => x.flip = flip,
            LeafletClassification::FromNdx(x) => x.flip = flip,
            LeafletClassification::Clustering(x) => x.flip = flip,
            LeafletClassification::SphericalClustering(x) => x.flip = flip,
        }

        self
    }

    /// Get the frequency of the analysis.
    #[inline(always)]
    pub fn get_frequency(&self) -> Frequency {
        match self {
            LeafletClassification::Global(x) => x.frequency(),
            LeafletClassification::Local(x) => x.frequency(),
            LeafletClassification::Individual(x) => x.frequency(),
            LeafletClassification::FromFile(x) => x.frequency(),
            LeafletClassification::FromMap(x) => x.frequency(),
            LeafletClassification::FromNdx(x) => x.frequency(),
            LeafletClassification::Clustering(x) => x.frequency(),
            LeafletClassification::SphericalClustering(x) => x.frequency(),
        }
    }

    /// Get the membrane normal specified for the leaflet classification.
    /// Returns `None` for Manual leaflet assignment.
    #[inline(always)]
    pub fn get_membrane_normal(&self) -> Option<Axis> {
        match self {
            LeafletClassification::Global(x) => x.membrane_normal(),
            LeafletClassification::Local(x) => x.membrane_normal(),
            LeafletClassification::Individual(x) => x.membrane_normal(),
            LeafletClassification::FromFile(_)
            | LeafletClassification::FromMap(_)
            | LeafletClassification::FromNdx(_) => None,
            LeafletClassification::Clustering(_)
            | LeafletClassification::SphericalClustering(_) => None,
        }
    }

    /// Should the leaflet classification data be collected and where?
    /// Always returns `Collect::Boolean(false)` for manual leaflet classification.
    #[inline(always)]
    pub fn get_collect(&self) -> &Collect {
        match self {
            LeafletClassification::Global(x) => x.collect(),
            LeafletClassification::Local(x) => x.collect(),
            LeafletClassification::Individual(x) => x.collect(),
            LeafletClassification::Clustering(x) => x.collect(),
            LeafletClassification::SphericalClustering(x) => x.collect(),
            LeafletClassification::FromFile(_)
            | LeafletClassification::FromMap(_)
            | LeafletClassification::FromNdx(_) => &Collect::Boolean(false),
        }
    }

    /// Should the leaflets be flipped.
    #[inline(always)]
    pub fn get_flip(&self) -> bool {
        match self {
            LeafletClassification::Global(x) => x.flip(),
            LeafletClassification::Local(x) => x.flip(),
            LeafletClassification::Individual(x) => x.flip(),
            LeafletClassification::FromFile(x) => x.flip(),
            LeafletClassification::FromMap(x) => x.flip(),
            LeafletClassification::FromNdx(x) => x.flip(),
            LeafletClassification::Clustering(x) => x.flip(),
            LeafletClassification::SphericalClustering(x) => x.flip(),
        }
    }

    /// Returns a radius of the cylinder for the calculation of local membrane center of geometry, if the method is Local.
    /// Otherwise, returns None.
    #[inline(always)]
    pub(crate) fn get_radius(&self) -> Option<f32> {
        match self {
            Self::Local(x) => Some(x.radius),
            _ => None,
        }
    }
}

/// Based on the global membrane center of geometry; useful for disrupted membranes; fast.
#[derive(Debug, Clone, Getters, CopyGetters, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GlobalParams {
    /// Selection of all lipids forming the membrane.
    #[getset(get = "pub")]
    membrane: String,
    /// Reference atoms identifying lipid headgroups (usually a phosphorus atom or a phosphate bead).
    /// There must only be one such atom/bead per lipid molecule.
    #[getset(get = "pub")]
    heads: String,
    /// Frequency of leaflet assignment.
    #[getset(get_copy = "pub")]
    #[serde(default)]
    frequency: Frequency,
    /// Orientation of the membrane normal.
    /// By default set based on the globally specified mmebrane normal.
    #[getset(get_copy = "pub")]
    #[serde(default)]
    membrane_normal: Option<Axis>,
    /// Should the leaflet assignment data be collected, stored and exported into an output file?
    #[getset(get = "pub")]
    #[serde(default, alias = "export")]
    collect: Collect,
    /// Will treat upper leaflet as lower leaflet and vice versa.
    #[getset(get_copy = "pub")]
    #[serde(default)] // false by default
    flip: bool,
}

/// Parameters for classification of lipids.
/// Based on the local membrane center of geometry; useful for curved membranes; slow.
#[derive(Debug, Clone, Getters, CopyGetters, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LocalParams {
    /// Selection of all lipids forming the membrane.
    #[getset(get = "pub")]
    membrane: String,
    /// Reference atoms identifying lipid headgroups (usually a phosphorus atom or a phosphate bead).
    /// There must only be one such atom/bead per lipid molecule.
    #[getset(get = "pub")]
    heads: String,
    /// Radius of a cylinder for the calculation of local membrane center of geometry (in nm).
    #[getset(get_copy = "pub")]
    #[serde(deserialize_with = "validate_radius")]
    radius: f32,
    /// Frequency of leaflet assignment.
    #[getset(get_copy = "pub")]
    #[serde(default)]
    frequency: Frequency,
    /// Orientation of the membrane normal.
    /// By default set based on the globally specified mmebrane normal.
    #[getset(get_copy = "pub")]
    #[serde(default)]
    membrane_normal: Option<Axis>,
    /// Should the leaflet assignment data be collected, stored and exported into an output file?
    #[getset(get = "pub")]
    #[serde(default, alias = "export")]
    collect: Collect,
    /// Will treat upper leaflet as lower leaflet and vice versa.
    #[getset(get_copy = "pub")]
    #[serde(default)] // false by default
    flip: bool,
}

fn validate_radius<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    let radius = f32::deserialize(deserializer)?;
    if radius <= 0.0 {
        Err(de::Error::custom("radius must be greater than 0"))
    } else {
        Ok(radius)
    }
}

/// Parameters for classification of lipids.
/// Based on the orientation of the lipid tails; less reliable; fast.
#[derive(Debug, Clone, Getters, CopyGetters, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IndividualParams {
    /// Reference atoms identifying lipid headgroups (usually a phosphorus atom or a phosphate bead).
    /// There must only be one such atom/bead per lipid molecule.
    #[getset(get = "pub")]
    heads: String,
    /// Reference atoms identifying methyl groups of lipid tails, i.e., the ends of lipid tails.
    /// There should be only one such atom/bead per one acyl chain in the molecule (e.g., two for lipids with two acyl chains).
    #[getset(get = "pub")]
    methyls: String,
    /// Frequency of leaflet assignment.
    #[getset(get_copy = "pub")]
    #[serde(default)]
    frequency: Frequency,
    /// Orientation of the membrane normal.
    /// By default set based on the globally specified mmebrane normal.
    #[getset(get_copy = "pub")]
    #[serde(default)]
    membrane_normal: Option<Axis>,
    /// Should the leaflet assignment data be collected, stored and exported into an output file?
    #[getset(get = "pub")]
    #[serde(default, alias = "export")]
    collect: Collect,
    /// Will treat upper leaflet as lower leaflet and vice versa.
    #[getset(get_copy = "pub")]
    #[serde(default)] // false by default
    flip: bool,
}

/// Classification of lipids into leaflets should be read from a separate leaflet assignment file.
#[derive(Debug, Clone, Getters, CopyGetters, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FromFileParams {
    /// Leaflet assignment file to read.
    #[getset(get = "pub")]
    file: String,
    /// Frequency of leaflet assignment.
    #[getset(get_copy = "pub")]
    #[serde(default)]
    frequency: Frequency,
    /// Will treat upper leaflet as lower leaflet and vice versa.
    #[getset(get_copy = "pub")]
    #[serde(default)] // false by default
    flip: bool,
}

impl<'de> Deserialize<'de> for FromFileParams {
    fn deserialize<D>(deser: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = Value::deserialize(deser)?;
        if let Value::String(s) = v {
            Ok(Self {
                file: s,
                frequency: Default::default(),
                flip: Default::default(),
            })
        } else {
            #[derive(Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Helper {
                file: String,
                #[serde(default)]
                frequency: Frequency,
                #[serde(default)]
                flip: bool,
            }
            let h: Helper = serde_yaml::from_value(v).map_err(serde::de::Error::custom)?;
            Ok(Self {
                file: h.file,
                frequency: h.frequency,
                flip: h.flip,
            })
        }
    }
}

/// Classification of lipids into leaflets should be read from a leaflet assignment hash map.
#[derive(Debug, Clone, Serialize, Getters, CopyGetters)]
#[serde(deny_unknown_fields)]
pub struct FromMapParams {
    /// Leaflet assignment map.
    #[getset(get = "pub")]
    assignment: HashMap<String, Vec<Vec<Leaflet>>>,
    /// Frequency of leaflet assignment.
    #[serde(default)]
    #[getset(get_copy = "pub")]
    frequency: Frequency,
    /// Will treat upper leaflet as lower leaflet and vice versa.
    #[getset(get_copy = "pub")]
    #[serde(default)] // false by default
    flip: bool,
}

impl<'de> Deserialize<'de> for FromMapParams {
    fn deserialize<D>(deser: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct MapVisitor;
        impl<'de> Visitor<'de> for MapVisitor {
            type Value = FromMapParams;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a mapping for FromMapParams")
            }
            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut assignment: Option<HashMap<String, Vec<Vec<Leaflet>>>> = None;
                let mut frequency: Option<Frequency> = None;
                let mut flip: Option<bool> = None;
                let mut raw = HashMap::new();
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "assignment" => {
                            if assignment.is_some() {
                                return Err(de::Error::duplicate_field("assignment"));
                            }
                            assignment = Some(map.next_value()?);
                        }
                        "frequency" => {
                            if frequency.is_some() {
                                return Err(de::Error::duplicate_field("frequency"));
                            }
                            frequency = Some(map.next_value()?);
                        }
                        "flip" => {
                            if flip.is_some() {
                                return Err(de::Error::duplicate_field("flip"));
                            }
                            flip = Some(map.next_value()?);
                        }
                        other => {
                            raw.insert(other.to_owned(), map.next_value()?);
                        }
                    }
                }
                Ok(FromMapParams {
                    assignment: assignment.unwrap_or(raw),
                    frequency: frequency.unwrap_or_default(),
                    flip: flip.unwrap_or_default(),
                })
            }
        }
        deser.deserialize_map(MapVisitor)
    }
}

/// Classification of lipids into leaflets should be read from one or multiple NDX files.
#[derive(Debug, Clone, Serialize, Deserialize, Getters, CopyGetters)]
#[serde(deny_unknown_fields)]
pub struct FromNdxParams {
    /// Head group atoms of the molecules. Only one head group atom per molecule!
    #[getset(get = "pub")]
    heads: String,
    /// One or more NDX files containing the groups defining leaflets.
    /// You can also use `glob` pattern when specifying the NDX files.
    #[getset(get = "pub")]
    #[serde(deserialize_with = "deserialize_string_or_vec")]
    ndx: Vec<String>,
    /// Name of the group identifying upper leaflet molecules.
    #[getset(get = "pub")]
    upper_leaflet: String,
    /// Name of the group identifying lower leaflet molecules.
    #[getset(get = "pub")]
    lower_leaflet: String,
    /// Frequency of leaflet assignment.
    #[serde(default)]
    #[getset(get_copy = "pub")]
    frequency: Frequency,
    /// Will treat upper leaflet as lower leaflet and vice versa.
    #[getset(get_copy = "pub")]
    #[serde(default)] // false by default
    flip: bool,
}

impl FromNdxParams {
    /// Print some of the NDX files that will be read.
    pub(crate) fn compact_display_ndx(&self) -> Result<String, std::fmt::Error> {
        let mut f = String::new();

        let n_ndx = self.ndx.len();
        if n_ndx == 1 {
            write!(f, "\nUsing an ndx file")?;
        } else {
            write!(f, "\nUsing ndx files:")?;
        }

        write!(f, " '{}'", self.ndx[0].cyan())?;

        if n_ndx <= 6 {
            for item in &self.ndx[1..] {
                write!(f, ", '{}'", item.cyan())?;
            }
        } else {
            for item in &self.ndx[1..3] {
                write!(f, ", '{}'", item.cyan())?;
            }
            write!(f, "{}", ", ...".clear())?;
            for item in &self.ndx[self.ndx.len() - 3..] {
                write!(f, ", '{}'", item.cyan())?;
            }
        }

        write!(f, "{}", ".".clear())?;

        Ok(f)
    }
}

pub fn deserialize_string_or_vec<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct StringOrVecVisitor;

    impl<'de> Visitor<'de> for StringOrVecVisitor {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a string or a sequence of strings")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            match glob::glob(value) {
                Ok(paths) => {
                    let files: Vec<_> = paths
                        .filter_map(Result::ok)
                        .map(|path| path.to_string_lossy().into_owned())
                        .collect();

                    if files.is_empty() {
                        Ok(vec![value.to_owned()])
                    } else {
                        Ok(files)
                    }
                }
                Err(e) => Err(de::Error::custom(e.to_string())),
            }
        }

        fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }
    }

    deserializer.deserialize_any(StringOrVecVisitor)
}

/// Using the spectral clustering. The only automatic method that can be used for membranes with arbitrary curved geometry.
#[derive(Debug, Clone, Getters, CopyGetters, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClusteringParams {
    /// Reference atoms identifying lipid headgroups (usually a phosphorus atom or a phosphate bead).
    /// There must only be one such atom/bead per lipid molecule.
    #[getset(get = "pub")]
    heads: String,
    /// Frequency of leaflet assignment.
    #[getset(get_copy = "pub")]
    #[serde(default)]
    frequency: Frequency,
    /// Should the leaflet assignment data be collected, stored and exported into an output file?
    #[getset(get = "pub")]
    #[serde(default, alias = "export")]
    collect: Collect,
    /// Will treat upper leaflet as lower leaflet and vice versa.
    #[getset(get_copy = "pub")]
    #[serde(default)] // false by default
    flip: bool,
}

#[derive(Debug, Clone, Getters, CopyGetters, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SphericalClusteringParams {
    /// Reference atoms identifying lipid headgroups (usually a phosphorus atom or a phosphate bead).
    /// There must only be one such atom/bead per lipid molecule.
    #[getset(get = "pub")]
    heads: String,
    /// Frequency of leaflet assignment.
    #[getset(get_copy = "pub")]
    #[serde(default)]
    frequency: Frequency,
    /// Should the leaflet assignment data be collected, stored and exported into an output file?
    #[getset(get = "pub")]
    #[serde(default, alias = "export")]
    collect: Collect,
    /// Will treat upper leaflet as lower leaflet and vice versa.
    #[getset(get_copy = "pub")]
    #[serde(default)] // false by default
    flip: bool,
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn compare_as_paths(path1: &str, path2: &str) {
        assert_eq!(Path::new(path1), Path::new(path2));
    }

    #[test]
    #[should_panic]
    fn local_radius_negative_fail() {
        let _classification = LeafletClassification::local("@membrane", "name P", -1.4);
    }

    #[test]
    #[should_panic]
    fn local_radius_zero_fail() {
        let _classification = LeafletClassification::local("@membrane", "name P", 0.0);
    }

    #[test]
    fn test_parse_manual_file_only() {
        let string = "!FromFile \"leaflets.yaml\"";
        match serde_yaml::from_str(string).unwrap() {
            LeafletClassification::FromFile(params) => {
                assert_eq!(params.file, "leaflets.yaml");
                assert_eq!(params.frequency, Frequency::every(1).unwrap());
            }
            _ => panic!("Invalid leaflet classification returned."),
        }

        let string = "!FromFile leaflets.yaml";
        match serde_yaml::from_str(string).unwrap() {
            LeafletClassification::FromFile(params) => {
                assert_eq!(params.file, "leaflets.yaml");
                assert_eq!(params.frequency, Frequency::every(1).unwrap());
            }
            _ => panic!("Invalid leaflet classification returned."),
        }
    }

    #[test]
    fn test_parse_manual_file_explicit() {
        let string = "!FromFile { file: leaflets.yaml }";
        match serde_yaml::from_str(string).unwrap() {
            LeafletClassification::FromFile(params) => {
                assert_eq!(params.file, "leaflets.yaml");
                assert_eq!(params.frequency, Frequency::every(1).unwrap());
            }
            _ => panic!("Invalid leaflet classification returned."),
        }
    }

    #[test]
    fn test_parse_manual_file_full() {
        let string = "!FromFile { file: leaflets.yaml, frequency: !Once }";
        match serde_yaml::from_str(string).unwrap() {
            LeafletClassification::FromFile(params) => {
                assert_eq!(params.file, "leaflets.yaml");
                assert_eq!(params.frequency, Frequency::once());
            }
            _ => panic!("Invalid leaflet classification returned."),
        }
    }

    fn compare_assignment(assignment: &HashMap<String, Vec<Vec<Leaflet>>>) {
        let expected_popc = [
            [
                Leaflet::Upper,
                Leaflet::Upper,
                Leaflet::Upper,
                Leaflet::Lower,
                Leaflet::Lower,
                Leaflet::Lower,
            ],
            [
                Leaflet::Upper,
                Leaflet::Lower,
                Leaflet::Upper,
                Leaflet::Upper,
                Leaflet::Lower,
                Leaflet::Upper,
            ],
        ];

        let expected_pope = [
            [
                Leaflet::Lower,
                Leaflet::Lower,
                Leaflet::Upper,
                Leaflet::Lower,
                Leaflet::Lower,
                Leaflet::Upper,
            ],
            [
                Leaflet::Upper,
                Leaflet::Lower,
                Leaflet::Lower,
                Leaflet::Upper,
                Leaflet::Lower,
                Leaflet::Lower,
            ],
        ];

        for (mol_type, expected) in ["POPC", "POPE"]
            .into_iter()
            .zip([expected_popc, expected_pope].into_iter())
        {
            let data = assignment.get(mol_type).unwrap();
            assert_eq!(data.len(), expected.len());
            for (frame, frame_expected) in data.iter().zip(expected.iter()) {
                assert_eq!(frame.len(), frame_expected.len());
                for (mol, mol_expected) in frame.iter().zip(frame_expected.iter()) {
                    assert_eq!(mol, mol_expected);
                }
            }
        }
    }

    #[test]
    fn test_parse_manual_map_only() {
        let string = "!Inline
POPC: [[1, 1, 1, 0, 0, 0], [1, 0, 1, 1, 0, 1]]
POPE:
  - [Lower, Lower, Upper, Lower, Lower, Upper]
  - [1, 0, 0, 1, 0, 0]";

        match serde_yaml::from_str(string).unwrap() {
            LeafletClassification::FromMap(params) => {
                compare_assignment(&params.assignment);
                assert_eq!(params.frequency, Frequency::every(1).unwrap());
            }
            _ => panic!("Invalid leaflet classification returned."),
        }
    }

    #[test]
    fn test_parse_manual_map_explicit() {
        let string = "!Inline
assignment:
  POPC: [[1, 1, 1, 0, 0, 0], [1, 0, 1, 1, 0, 1]]
  POPE:
    - [Lower, Lower, Upper, Lower, Lower, Upper]
    - [1, 0, 0, 1, 0, 0]";

        match serde_yaml::from_str(string).unwrap() {
            LeafletClassification::FromMap(params) => {
                compare_assignment(&params.assignment);
                assert_eq!(params.frequency, Frequency::every(1).unwrap());
            }
            _ => panic!("Invalid leaflet classification returned."),
        }
    }

    #[test]
    fn test_parse_manual_map_full() {
        let string = "!FromMap
assignment:
  POPC: [[1, 1, 1, 0, 0, 0], [1, 0, 1, 1, 0, 1]]
  POPE:
    - [Lower, Lower, Upper, Lower, Lower, Upper]
    - [1, 0, 0, 1, 0, 0]
frequency: !Once";

        match serde_yaml::from_str(string).unwrap() {
            LeafletClassification::FromMap(params) => {
                compare_assignment(&params.assignment);
                assert_eq!(params.frequency, Frequency::once());
            }
            _ => panic!("Invalid leaflet classification returned."),
        }
    }

    #[test]
    fn test_parse_manual_ndx_single() {
        let string = "!FromNdx
heads: \"name P\"
ndx: tests/files/cg.ndx
upper_leaflet: UpperLeaflet
lower_leaflet: LowerLeaflet";

        match serde_yaml::from_str(string).unwrap() {
            LeafletClassification::FromNdx(params) => {
                assert_eq!(params.heads(), "name P");
                compare_as_paths(&params.ndx()[0], "tests/files/cg.ndx");
                assert_eq!(params.upper_leaflet(), "UpperLeaflet");
                assert_eq!(params.lower_leaflet(), "LowerLeaflet");
                assert_eq!(params.frequency(), Frequency::every(1).unwrap());
            }
            _ => panic!("Invalid leaflet classification returned."),
        }
    }

    #[test]
    fn test_parse_manual_ndx_single_frequency() {
        let string = "!FromNdx
heads: \"name P\"
ndx: tests/files/pcpepg.ndx
frequency: !Once
upper_leaflet: UpperLeaflet
lower_leaflet: LowerLeaflet";

        match serde_yaml::from_str(string).unwrap() {
            LeafletClassification::FromNdx(params) => {
                assert_eq!(params.heads(), "name P");
                compare_as_paths(&params.ndx()[0], "tests/files/pcpepg.ndx");
                assert_eq!(params.upper_leaflet(), "UpperLeaflet");
                assert_eq!(params.lower_leaflet(), "LowerLeaflet");
                assert_eq!(params.frequency(), Frequency::once());
            }
            _ => panic!("Invalid leaflet classification returned."),
        }
    }

    #[test]
    fn test_parse_manual_ndx_multiple() {
        let string = "!FromNdx
heads: \"name P\"
ndx: [index1.ndx, index2.ndx, index3.ndx]
upper_leaflet: UpperLeaflet
lower_leaflet: LowerLeaflet";

        match serde_yaml::from_str(string).unwrap() {
            LeafletClassification::FromNdx(params) => {
                assert_eq!(params.heads(), "name P");
                assert_eq!(
                    params.ndx(),
                    &vec!["index1.ndx", "index2.ndx", "index3.ndx"]
                );
                assert_eq!(params.upper_leaflet(), "UpperLeaflet");
                assert_eq!(params.lower_leaflet(), "LowerLeaflet");
                assert_eq!(params.frequency(), Frequency::every(1).unwrap());
            }
            _ => panic!("Invalid leaflet classification returned."),
        }
    }

    #[test]
    fn test_parse_manual_ndx_multiple_frequency() {
        let string = "!FromNdx
heads: \"name P\"
ndx: [index1.ndx, index2.ndx, index3.ndx]
upper_leaflet: UpperLeaflet
frequency: !Every 5
lower_leaflet: LowerLeaflet";

        match serde_yaml::from_str(string).unwrap() {
            LeafletClassification::FromNdx(params) => {
                assert_eq!(params.heads(), "name P");
                assert_eq!(
                    params.ndx(),
                    &vec!["index1.ndx", "index2.ndx", "index3.ndx"]
                );
                assert_eq!(params.upper_leaflet(), "UpperLeaflet");
                assert_eq!(params.lower_leaflet(), "LowerLeaflet");
                assert_eq!(params.frequency(), Frequency::every(5).unwrap());
            }
            _ => panic!("Invalid leaflet classification returned."),
        }
    }

    #[test]
    fn test_parse_manual_ndx_glob_multiple() {
        let string = "!FromNdx
heads: \"name P\"
ndx: \"tests/files/ndx/glob/index*.ndx\"
upper_leaflet: UpperLeaflet
lower_leaflet: LowerLeaflet";

        match serde_yaml::from_str(string).unwrap() {
            LeafletClassification::FromNdx(params) => {
                assert_eq!(params.heads(), "name P");
                compare_as_paths(&params.ndx()[0], "tests/files/ndx/glob/index1.ndx");
                compare_as_paths(&params.ndx()[1], "tests/files/ndx/glob/index2.ndx");
                assert_eq!(params.upper_leaflet(), "UpperLeaflet");
                assert_eq!(params.lower_leaflet(), "LowerLeaflet");
                assert_eq!(params.frequency(), Frequency::every(1).unwrap());
            }
            _ => panic!("Invalid leaflet classification returned."),
        }
    }

    #[test]
    fn test_specifying_collect() {
        let classification =
            LeafletClassification::global("@membrane", "name P").with_collect(true);
        assert_eq!(classification.get_collect(), &Collect::Boolean(true));

        let classification =
            LeafletClassification::local("@membrane", "name P", 2.5).with_collect(false);
        assert_eq!(classification.get_collect(), &Collect::Boolean(false));

        let classification =
            LeafletClassification::clustering("name P").with_collect("leaflets.yaml");
        assert_eq!(
            classification.get_collect(),
            &Collect::File(String::from("leaflets.yaml"))
        );
    }

    #[test]
    fn test_parse_collect() {
        for string in [
            "!Global
membrane: \"@membrane\"
heads: \"name P\"
collect: \"leaflets.yaml\"",
            "!Global
membrane: \"@membrane\"
heads: \"name P\"
export: \"leaflets.yaml\"",
        ] {
            match serde_yaml::from_str(string).unwrap() {
                LeafletClassification::Global(params) => {
                    assert_eq!(params.membrane(), "@membrane");
                    assert_eq!(params.heads(), "name P");
                    assert_eq!(
                        params.collect(),
                        &Collect::File(String::from("leaflets.yaml"))
                    );
                }
                _ => panic!("Invalid leaflet classification returned."),
            }
        }

        for string in [
            "!Local
membrane: \"@membrane\"
heads: \"name P\"
radius: 2.5
collect: \"leaflets.yaml\"",
            "!Local
membrane: \"@membrane\"
heads: \"name P\"
radius: 2.5
export: \"leaflets.yaml\"",
        ] {
            match serde_yaml::from_str(string).unwrap() {
                LeafletClassification::Local(params) => {
                    assert_eq!(params.membrane(), "@membrane");
                    assert_eq!(params.heads(), "name P");
                    assert_eq!(params.radius(), 2.5);
                    assert_eq!(
                        params.collect(),
                        &Collect::File(String::from("leaflets.yaml"))
                    );
                }
                _ => panic!("Invalid leaflet classification returned."),
            }
        }

        for string in [
            "!Individual
heads: \"name P\"
methyls: \"name C316 C218\"
collect: \"leaflets.yaml\"",
            "!Individual
heads: \"name P\"
methyls: \"name C316 C218\"
export: \"leaflets.yaml\"",
        ] {
            match serde_yaml::from_str(string).unwrap() {
                LeafletClassification::Individual(params) => {
                    assert_eq!(params.heads(), "name P");
                    assert_eq!(params.methyls(), "name C316 C218");
                    assert_eq!(
                        params.collect(),
                        &Collect::File(String::from("leaflets.yaml"))
                    );
                }
                _ => panic!("Invalid leaflet classification returned."),
            }
        }

        for string in [
            "!Clustering
heads: \"name P\"
collect: \"leaflets.yaml\"",
            "!Clustering
heads: \"name P\"
export: \"leaflets.yaml\"",
        ] {
            match serde_yaml::from_str(string).unwrap() {
                LeafletClassification::Clustering(params) => {
                    assert_eq!(params.heads(), "name P");
                    assert_eq!(
                        params.collect(),
                        &Collect::File(String::from("leaflets.yaml"))
                    );
                }
                _ => panic!("Invalid leaflet classification returned."),
            }
        }

        for string in [
            "!SphericalClustering
heads: \"name P\"
collect: \"leaflets.yaml\"",
            "!SphericalClustering
heads: \"name P\"
export: \"leaflets.yaml\"",
        ] {
            match serde_yaml::from_str(string).unwrap() {
                LeafletClassification::SphericalClustering(params) => {
                    assert_eq!(params.heads(), "name P");
                    assert_eq!(
                        params.collect(),
                        &Collect::File(String::from("leaflets.yaml"))
                    );
                }
                _ => panic!("Invalid leaflet classification returned."),
            }
        }
    }

    #[test]
    fn test_parse_flip() {
        let string = "!Global
  membrane: \"@membrane\"
  heads: \"name P\"
  flip: true";

        match serde_yaml::from_str(string).unwrap() {
            LeafletClassification::Global(params) => {
                assert_eq!(params.membrane(), "@membrane");
                assert_eq!(params.heads(), "name P");
                assert!(params.flip())
            }
            _ => panic!("Invalid leaflet classification returned."),
        }

        let string = "!Local
  membrane: \"@membrane\"
  heads: \"name P\"
  radius: 2.5
  flip: true";

        match serde_yaml::from_str(string).unwrap() {
            LeafletClassification::Local(params) => {
                assert_eq!(params.membrane(), "@membrane");
                assert_eq!(params.heads(), "name P");
                assert!(params.flip())
            }
            _ => panic!("Invalid leaflet classification returned."),
        }

        let string = "!Individual
  heads: \"name P\"
  methyls: \"name C218 C316\"
  flip: true";

        match serde_yaml::from_str(string).unwrap() {
            LeafletClassification::Individual(params) => {
                assert_eq!(params.heads(), "name P");
                assert_eq!(params.methyls(), "name C218 C316");
                assert!(params.flip())
            }
            _ => panic!("Invalid leaflet classification returned."),
        }

        let string = "!Clustering
  heads: \"name P\"
  flip: true";

        match serde_yaml::from_str(string).unwrap() {
            LeafletClassification::Clustering(params) => {
                assert_eq!(params.heads(), "name P");
                assert!(params.flip())
            }
            _ => panic!("Invalid leaflet classification returned."),
        }

        let string = "!SphericalClustering
  heads: \"name P\"
  flip: true";

        match serde_yaml::from_str(string).unwrap() {
            LeafletClassification::SphericalClustering(params) => {
                assert_eq!(params.heads(), "name P");
                assert!(params.flip())
            }
            _ => panic!("Invalid leaflet classification returned."),
        }

        let string = "!FromFile
  file: leaflets.yaml
  flip: true";

        match serde_yaml::from_str(string).unwrap() {
            LeafletClassification::FromFile(params) => {
                assert_eq!(params.file(), "leaflets.yaml");
                assert!(params.flip())
            }
            _ => panic!("Invalid leaflet classification returned."),
        }

        let string = "!Inline
  assignment:
    POPE:
      - [Upper, Upper, Upper, Upper, Upper, Upper, Upper, Upper, Upper, Upper, Upper, Upper]
    POPC:
      - [Lower, Lower, Lower, Lower, Lower, Lower, Lower, Lower, Lower, Lower]
    POPG:
      - [Upper, Upper, Upper, Upper, Lower, Lower, Lower, Lower]
  flip: true";

        match serde_yaml::from_str(string).unwrap() {
            LeafletClassification::FromMap(params) => {
                assert!(params.assignment().contains_key("POPE"));
                assert!(params.assignment().contains_key("POPC"));
                assert!(params.assignment().contains_key("POPG"));
                assert!(params.flip())
            }
            _ => panic!("Invalid leaflet classification returned."),
        }

        let string = "!FromNdx
  ndx: \"leaflets.ndx\"
  heads: \"name P\"
  upper_leaflet: \"UpperLeaflet\"
  lower_leaflet: \"LowerLeaflet\"
  flip: true";

        match serde_yaml::from_str(string).unwrap() {
            LeafletClassification::FromNdx(params) => {
                assert_eq!(params.heads(), "name P");
                assert_eq!(params.ndx(), &vec![String::from("leaflets.ndx")]);
                assert!(params.flip())
            }
            _ => panic!("Invalid leaflet classification returned."),
        }
    }

    #[test]
    fn test_parse_collect_manual_fail() {
        let string = "!FromFile
file: \"input_leaflets.yaml\"
collect: \"leaflets.yaml\"";

        let params: Result<LeafletClassification, _> = serde_yaml::from_str(string);
        assert!(params.is_err());

        let string = "!Inline
assignment: 
    POPE:
      - [Upper, Upper, Upper, Upper, Upper, Upper, Upper, Upper, Upper, Upper, Upper, Upper]
    POPC:
      - [Lower, Lower, Lower, Lower, Lower, Lower, Lower, Lower, Lower, Lower]
    POPG:
      - [Upper, Upper, Upper, Upper, Lower, Lower, Lower, Lower]
frequency: !Once
export: \"leaflets.yaml\"";

        let params: Result<LeafletClassification, _> = serde_yaml::from_str(string);
        assert!(params.is_err());

        let string = "!FromNdx
  ndx: \"leaflets.ndx\"
  heads: \"name P\"
  upper_leaflet: \"UpperLeaflet\"
  lower_leaflet: \"LowerLeaflet\"
  collect: \"leaflets.yaml\"";

        let params: Result<LeafletClassification, _> = serde_yaml::from_str(string);
        assert!(params.is_err());
    }

    #[test]
    fn test_parse_manual_fail_1() {
        let string = "!FromFile 7";
        let params: Result<LeafletClassification, _> = serde_yaml::from_str(string);
        assert!(params.is_err());
    }

    #[test]
    fn test_parse_manual_fail_2() {
        let string = "!Inline
POPC: [[1, 1, 1, 0, 0, 0], [1, 0, 1, 1, 0, 1]]
POPE:
  - [Lower, Lower, Upp3r, Lower, Lower, Upper]
  - [1, 0, 0, 1, 0, 0]";
        let params: Result<LeafletClassification, _> = serde_yaml::from_str(string);
        assert!(params.is_err());
    }

    #[test]
    fn test_parse_manual_fail_3() {
        let string = "!Inline
POPC: [[1, 1, -1, 0, 0, 0], [1, 0, 1, 1, 0, 1]]
POPE:
  - [Lower, Lower, Upper, Lower, Lower, Upper]
  - [1, 0, 0, 1, 0, 0]";
        let params: Result<LeafletClassification, _> = serde_yaml::from_str(string);
        assert!(params.is_err());
    }

    #[test]
    fn test_parse_manual_fail_4() {
        let string = "!Inline
POPC: leaflets.yaml
POPE:
  - [Lower, Lower, Upper, Lower, Lower, Upper]
  - [1, 0, 0, 1, 0, 0]";
        let params: Result<LeafletClassification, _> = serde_yaml::from_str(string);
        assert!(params.is_err());
    }

    #[test]
    fn test_parse_manual_fail_5() {
        let string = "!Inline
assignment: leaflets.yaml
frequency: !Once";

        let params: Result<LeafletClassification, _> = serde_yaml::from_str(string);
        assert!(params.is_err());
    }

    #[test]
    fn test_parse_manual_fail_6() {
        let string = "!FromFile
file: leaflets.yaml
frequency: some";

        let params: Result<LeafletClassification, _> = serde_yaml::from_str(string);
        assert!(params.is_err());
    }

    #[test]
    fn test_parse_local_negative_radius_fail() {
        let string = "!Local
membrane: \"@membrane\"
heads: name P
radius: -1.5";

        let result: Result<LeafletClassification, _> = serde_yaml::from_str(string);

        match result {
            Err(e) => assert_eq!(e.to_string(), "radius must be greater than 0"),
            Ok(_) => panic!("Should have failed."),
        }
    }

    #[test]
    fn test_parse_local_zero_radius_fail() {
        let string = "!Local
membrane: \"@membrane\"
heads: name P
radius: 0.0";

        let result: Result<LeafletClassification, _> = serde_yaml::from_str(string);

        match result {
            Err(e) => assert_eq!(e.to_string(), "radius must be greater than 0"),
            Ok(_) => panic!("Should have failed."),
        }
    }
}
