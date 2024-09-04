use crate::build_tree::build_tree;
use crate::state::merge_states::MergeStates;
use crate::state::min_max_mean::MinMaxMeanState;
use crate::state::value::ValueState;
use crate::state::{min_max_mean, value};
use crate::tree::Tree;
use crate::tree_config::TreeConfig;
use numpy::ndarray::ArrayView1 as NdArrayView;
use numpy::PyArrayDescr;
use numpy::{PyArray1, PyUntypedArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::convert::Infallible;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

pyo3::import_exception!(pickle, PicklingError);
pyo3::import_exception!(pickle, UnpicklingError);

/// A merger algorithm for MOMBuilder.
///
/// Parameters
/// ----------
/// state : str
///     State type. It must be one of the following:
///     - "min-max-mean" for MinMaxMeanState
///     - "value" for ValueState
/// merger : str
///     Merger algorithm. It must be one of the following:
///     - For "min-max-mean" state:
///       - "rtol" - merger would check the relative difference between
///         minimum and maximum values, and merge the states if it is less than
///         the threshold. The threshold is specified with "threshold" kwarg.
///     - For "value" state:
///       - "equal" - merger would merge the states if all values are exactly
///         equal.
/// dtype : numpy.dtype
///     Data type of the input arrays.
/// **kwargs : dict
///     Additional arguments for the merger, please see the `merger` parameter
///     description.
#[derive(Clone, Serialize, Deserialize)]
#[pyclass(name = "MOMMerger", module = "mom_builder.mom_builder")]
struct MomMerger {
    tree_states: PyStates,
    dtype_char: char,
}

#[pymethods]
impl MomMerger {
    #[new]
    #[pyo3(signature = (state, merger, *, dtype, **kwargs))]
    fn __new__(
        state: &str,
        merger: &str,
        dtype: &PyArrayDescr,
        kwargs: Option<HashMap<&str, f64>>,
    ) -> PyResult<Self> {
        let kwargs = kwargs.unwrap_or_default();

        let dtype_char = dtype.char() as char;

        let tree_states = match state.to_lowercase().replace("_", "-").as_str() {
            "min-max-mean" => match merger.to_lowercase().as_str() {
                "rtol" => {
                    if kwargs.keys().len() != 1 {
                        return Err(PyValueError::new_err(
                            "state='min-max-mean' and merger='rtol' require exactly one additional keyword argument: threshold",
                        ));
                    }
                    let threshold = *kwargs
                            .get("threshold")
                            .ok_or_else(|| PyValueError::new_err(r#"threshold keyword argument is required for state="min-max-mean" and merger="rtol""#))?;

                    match dtype_char {
                        'f' => {
                            let state_validator =
                                min_max_mean::RelativeToleranceValidator::new(threshold as f32);
                            let state_merger = min_max_mean::Merger::new(state_validator);
                            PyStates::MinMaxMean(PyMinMaxMeanStates::F32(GenericStates::new(
                                state_merger,
                            )))
                        }
                        'd' => {
                            let state_validator =
                                min_max_mean::RelativeToleranceValidator::new(threshold);
                            let state_merger = min_max_mean::Merger::new(state_validator);
                            PyStates::MinMaxMean(PyMinMaxMeanStates::F64(GenericStates::new(
                                state_merger,
                            )))
                        }
                        _ => {
                            return Err(PyValueError::new_err(
                                r#"state "min-max-mean" supports only float32 and float64 dtypes"#,
                            ))
                        }
                    }
                }
                _ => return Err(PyValueError::new_err("Unknown min-max-mean merger")),
            },
            "value" => match merger.to_lowercase().as_str() {
                "equal" => {
                    if !kwargs.is_empty() {
                        return Err(PyValueError::new_err(
                            "No keyword arguments are allowed for state='value' and merger='equal'",
                        ));
                    }

                    match dtype_char {
                        '?' => PyStates::Value(PyValueStates::BoolEqual(GenericStates::default())),
                        'b' => PyStates::Value(PyValueStates::I8Equal(GenericStates::default())),
                        'h' => PyStates::Value(PyValueStates::I16Equal(GenericStates::default())),
                        'i' => PyStates::Value(PyValueStates::I32Equal(GenericStates::default())),
                        'q' => PyStates::Value(PyValueStates::I64Equal(GenericStates::default())),
                        'B' => PyStates::Value(PyValueStates::U8Equal(GenericStates::default())),
                        'H' => PyStates::Value(PyValueStates::U16Equal(GenericStates::default())),
                        'I' => PyStates::Value(PyValueStates::U32Equal(GenericStates::default())),
                        'Q' => PyStates::Value(PyValueStates::U64Equal(GenericStates::default())),
                        'f' => PyStates::Value(PyValueStates::F32Equal(GenericStates::default())),
                        'd' => PyStates::Value(PyValueStates::F64Equal(GenericStates::default())),
                        _ => {
                            return Err(PyValueError::new_err(
                                r#"Only bool, integer and floating point dtypes are supported for state="value" and merger="equal""#,
                            ))
                        }
                    }
                }
                "sum-threshold" => {
                    if kwargs.keys().len() != 1 {
                        return Err(PyValueError::new_err(
                            "state='value' and merger='sum-threshold' require exactly one additional keyword argument: threshold",
                        ));
                    }
                    let threshold = *kwargs
                            .get("threshold")
                            .ok_or_else(|| PyValueError::new_err(r#"threshold keyword argument is required for state="value" and merger="sum-threshold""#))?;

                    match dtype_char {
                        'b' => PyStates::Value(PyValueStates::I8SumThreshold(GenericStates::new(
                            value::SumUntilMerger::new(value::MaximumValueValidator(
                                threshold as i8,
                            )),
                        ))),
                        _ => {
                            return Err(PyValueError::new_err(
                                r#"Only int8 dtype is supported for state="value" and merger="sum-threshold""#,
                            ))
                        }
                    }
                }
                _ => return Err(PyValueError::new_err("Unknown value merger")),
            },
            _ => return Err(PyValueError::new_err("Unknown state type")),
        };

        Ok(Self {
            tree_states,
            dtype_char,
        })
    }

    // pickle support

    fn __getnewargs_ex__(
        &self,
        py: Python,
    ) -> PyResult<(
        (&'static str, &'static str),
        HashMap<&'static str, PyObject>,
    )> {
        let state_str = match &self.tree_states {
            PyStates::MinMaxMean(_) => "min-max-mean",
            PyStates::Value(_) => "value",
        };
        let merger_str = match &self.tree_states {
            PyStates::MinMaxMean(_) => "rtol",
            PyStates::Value(_) => "equal",
        };
        let kwargs = {
            let mut kwargs = match &self.tree_states {
                PyStates::MinMaxMean(PyMinMaxMeanStates::F32(generic)) => [(
                    "threshold",
                    generic.merger.validator.threshold().into_py(py),
                )]
                .into_iter()
                .collect(),
                PyStates::MinMaxMean(PyMinMaxMeanStates::F64(generic)) => [(
                    "threshold",
                    generic.merger.validator.threshold().into_py(py),
                )]
                .into_iter()
                .collect(),
                PyStates::Value(_) => HashMap::new(),
            };
            kwargs.insert(
                "dtype",
                PyArrayDescr::new(py, &self.dtype_char)?.into_py(py),
            );
            kwargs
        };

        Ok(((state_str, merger_str), kwargs))
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> &'py PyBytes {
        PyBytes::new(py, &[])
    }

    fn __setstate__(&mut self, _state: &PyBytes) {
        ()
    }

    // copy/deepcopy support
    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: Py<PyAny>) -> Self {
        self.clone()
    }
}

/// A low-level two-step builder of multi-order healpix maps.
///
/// The builder is supposed to be used in the following way:
/// 1. Create a builder with `MOMBuilder` constructor.
/// 2. Build all subtrees with `build_subtree()` method.
///    Use `subtree_maxnorder_indexes()` method to get indexes of
///    the subtree leaves on the maximum depth.
/// 3. Build the top tree with `build_top_tree()` method.
///
///
/// Note, that dtype of the input arrays must be the same for all subtrees,
/// and all subtrees must be built before building the top tree.
/// Consider to user `gen_mom_from_fn()` function, which wraps this class
/// and provides a more convenient interface.
///
///   t   t                     0
///   o   r      _______________|_________________
///   p   e     /          |           |          \
///       e    0           1           2          3
///          / | | \    / | | \    / | | \   /  |  | \
///  sub-   0 1  2 3   4  5 6  7  8 9 10 11 12 13 14 15
///  trees |||||||||  |||||||||  |||||||||  |||||||||||
///  ...   xxxxxxxxx  xxxxxxxxx  xxxxxxxxx  xxxxxxxxxxx
///
/// (1/12 of a healpix tree. split_norder=1, max_norder>2. We first build four
///  subtrees, and then, if all four subtree roots are not empty and mergeable,
///  we build the top tree.)
///
/// Parameters
/// ----------
/// merger : MOMMerger
///     Merger algorithm to use to build the tree.
/// max_norder : int
///     Maximum depth of the healpix tree. It is the maximum norder of the
///     whole tree, not of the subtrees.
/// split_norder : int
///     Maximum depth of the top tree. It is level of the tree where it is
///     split into subtrees.
/// thread_safe : bool
///     If True, the builder will use a thread-safe implementation. It forces
///     copying of the input arrays to release GIL, so it could be a bit
///     slower. If False, the builder will use a non-thread-safe
///     implementation, which could make itself into a deadlock if used in
///     a multithreaded environment.
///
/// Attributes
/// ----------
/// max_norder : int
/// split_norder : int
/// num_subtrees : int
///     Number of subtrees, it is equal to the number of leaves in the top
///     tree on its maximum depth.
///
/// Methods
/// -------
/// subtree_maxnorder_indexes(subtree_index)
///     Returns indexes of the subtree leaves on the maximum depth.
/// build_subtree(subtree_index, a)
///     Builds a subtree from an array of leaf values. The array must
///     correspond to indexes given by `subtree_maxnorder_indexes()`.
///     Reurns a list of (norder, indexes, values) tuples.
/// build_top_tree()
///     Builds the top tree from subtrees. Returns a list of
///     (norder, indexes, values) tuples.
/// extend(other)
///     Extends the builder with another builder. That builder will be cleared
///
#[derive(Clone, Serialize, Deserialize)]
#[pyclass(name = "MOMBuilder", module = "mom_builder.mom_builder")]
struct MomBuilder {
    py_builder_config: PyBuilderConfig,
    states: PyStates,
}

#[derive(Clone, Serialize, Deserialize)]
struct PyBuilderConfig {
    split_norder: usize,
    max_norder: usize,
    thread_safe: bool,
    subtree_config: TreeConfig,
    top_tree_config: TreeConfig,
    mom_merger: MomMerger,
}

impl PyBuilderConfig {
    fn max_norder_index_offset(&self, subtree_index: usize) -> usize {
        subtree_index * self.subtree_config.max_norder_nleaves()
    }
}

#[derive(Clone, Serialize, Deserialize)]
enum PyStates {
    MinMaxMean(PyMinMaxMeanStates),
    Value(PyValueStates),
}

#[derive(Clone, Serialize, Deserialize)]
struct GenericStates<T, State, Merger>
where
    Merger: MergeStates<State = State>,
{
    phantom_element: PhantomData<T>,
    states: StateTree<State>,
    merger: Merger,
}

impl<T, State, Merger> GenericStates<T, State, Merger>
where
    Merger: MergeStates<State = State>,
{
    fn new(merger: Merger) -> Self {
        Self {
            phantom_element: PhantomData,
            states: StateTree::default(),
            merger,
        }
    }
}

impl<T> Default for GenericStates<T, ValueState<T>, value::ExactlyEqualMerger<T>>
where
    T: Copy + Default + PartialEq,
{
    fn default() -> Self {
        Self::new(value::ExactlyEqualMerger::default())
    }
}

type StateTree<State> = Arc<RwLock<BTreeMap<usize, Option<State>>>>;

#[derive(Clone, Serialize, Deserialize)]
enum PyMinMaxMeanStates {
    F32(
        GenericStates<
            f32,
            MinMaxMeanState<f32>,
            min_max_mean::Merger<min_max_mean::RelativeToleranceValidator<f32>>,
        >,
    ),
    F64(
        GenericStates<
            f64,
            MinMaxMeanState<f64>,
            min_max_mean::Merger<min_max_mean::RelativeToleranceValidator<f64>>,
        >,
    ),
}

#[derive(Clone, Serialize, Deserialize)]
enum PyValueStates {
    // Exact equality merger
    BoolEqual(GenericStates<bool, ValueState<bool>, value::ExactlyEqualMerger<bool>>),
    I8Equal(GenericStates<i8, ValueState<i8>, value::ExactlyEqualMerger<i8>>),
    I16Equal(GenericStates<i16, ValueState<i16>, value::ExactlyEqualMerger<i16>>),
    I32Equal(GenericStates<i32, ValueState<i32>, value::ExactlyEqualMerger<i32>>),
    I64Equal(GenericStates<i64, ValueState<i64>, value::ExactlyEqualMerger<i64>>),
    U8Equal(GenericStates<u8, ValueState<u8>, value::ExactlyEqualMerger<u8>>),
    U16Equal(GenericStates<u16, ValueState<u16>, value::ExactlyEqualMerger<u16>>),
    U32Equal(GenericStates<u32, ValueState<u32>, value::ExactlyEqualMerger<u32>>),
    U64Equal(GenericStates<u64, ValueState<u64>, value::ExactlyEqualMerger<u64>>),
    F32Equal(GenericStates<f32, ValueState<f32>, value::ExactlyEqualMerger<f32>>),
    F64Equal(GenericStates<f64, ValueState<f64>, value::ExactlyEqualMerger<f64>>),
    // Sum until threshold
    I8SumThreshold(
        GenericStates<i8, ValueState<i8>, value::SumUntilMerger<value::MaximumValueValidator<i8>>>,
    ),
}

#[pymethods]
impl MomBuilder {
    #[new]
    #[pyo3(signature = (merger, *, max_norder, split_norder, thread_safe=true))]
    fn __new__(
        merger: MomMerger,
        max_norder: usize,
        split_norder: usize,
        thread_safe: bool,
    ) -> Self {
        Self {
            py_builder_config: PyBuilderConfig {
                split_norder,
                max_norder,
                thread_safe,
                subtree_config: TreeConfig::new(1usize, 4usize, max_norder - split_norder),
                top_tree_config: TreeConfig::new(12usize, 4usize, split_norder),
                mom_merger: merger.clone(),
            },
            states: merger.tree_states,
        }
    }

    /// Maximum depth of the healpix tree.
    #[getter]
    fn max_norder(&self) -> usize {
        self.py_builder_config.max_norder
    }

    /// Depth of the top tree.
    #[getter]
    fn split_norder(&self) -> usize {
        self.py_builder_config.split_norder
    }

    /// Number of subtrees
    ///
    /// It is equal to the number of leaves in the top tree on its maximum
    /// depth.
    #[getter]
    fn num_subtrees(&self) -> usize {
        self.py_builder_config.top_tree_config.max_norder_nleaves()
    }

    /// MOMMerger instanced used in construction
    #[getter]
    fn merger(&self) -> MomMerger {
        self.py_builder_config.mom_merger.clone()
    }

    /// Returns healpix indexes of the subtree leaves on the maximum depth.
    ///
    /// It is supposed to be used to create an input array for
    /// `build_subtree()`.
    ///
    /// Parameters
    /// ----------
    /// subtree_index : int
    ///     Index of the subtree, in [0, num_subtrees).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of uint64
    ///     Array of healpix indexes of the subtree leaves on the maximum
    ///     norder.
    ///
    fn subtree_maxnorder_indexes<'py>(
        &self,
        py: Python<'py>,
        subtree_index: usize,
    ) -> PyResult<&'py PyArray1<usize>> {
        if subtree_index >= self.py_builder_config.top_tree_config.max_norder_nleaves() {
            return Err(PyValueError::new_err("subtree_index is out of range"));
        }
        let offset = self
            .py_builder_config
            .max_norder_index_offset(subtree_index);

        let output = PyArray1::from_vec(
            py,
            (offset..offset + self.py_builder_config.subtree_config.max_norder_nleaves()).collect(),
        );
        Ok(output)
    }

    /// Builds a subtree from an array of leaf values.
    ///
    /// This method builds a subtree, stores the root node state if exists
    /// and returns the rest of the subtree otherwise.
    ///
    /// It must be called once per subtree, and the subtrees must be built
    /// with `a` of the same dtype.
    ///
    /// Parameters
    /// ----------
    /// subtree_index : int
    ///     Healpix index of the subtree root, in [0, num_subtrees).
    /// a : numpy.ndarray of float32 or float64
    ///     Leaf values at the maximum healpix norder. It must correspond
    ///     to indexes given by `subtree_maxnorder_indexes()`.
    ///
    /// Returns
    /// -------
    /// list of (int, numpy.ndarray of uint64, numpy.ndarray of float32/64)
    ///     List of (norder, indexes, values) tuples, a tuple per non-empty
    ///     norder. The norder is absolute (not relative to the subtree).
    ///
    fn build_subtree<'py>(
        &self,
        py: Python<'py>,
        subtree_index: usize,
        a: &'py PyAny,
    ) -> PyResult<Vec<(usize, &'py PyArray1<usize>, &'py PyUntypedArray)>> {
        match &self.states {
            PyStates::MinMaxMean(PyMinMaxMeanStates::F32(generic)) => {
                generic.build_subtree(py, subtree_index, a.downcast()?, &self.py_builder_config)
            }
            PyStates::MinMaxMean(PyMinMaxMeanStates::F64(generic)) => {
                generic.build_subtree(py, subtree_index, a.downcast()?, &self.py_builder_config)
            }
            PyStates::Value(PyValueStates::BoolEqual(generic)) => {
                generic.build_subtree(py, subtree_index, a.downcast()?, &self.py_builder_config)
            }
            PyStates::Value(PyValueStates::I8Equal(generic)) => {
                generic.build_subtree(py, subtree_index, a.downcast()?, &self.py_builder_config)
            }
            PyStates::Value(PyValueStates::I16Equal(generic)) => {
                generic.build_subtree(py, subtree_index, a.downcast()?, &self.py_builder_config)
            }
            PyStates::Value(PyValueStates::I32Equal(generic)) => {
                generic.build_subtree(py, subtree_index, a.downcast()?, &self.py_builder_config)
            }
            PyStates::Value(PyValueStates::I64Equal(generic)) => {
                generic.build_subtree(py, subtree_index, a.downcast()?, &self.py_builder_config)
            }
            PyStates::Value(PyValueStates::U8Equal(generic)) => {
                generic.build_subtree(py, subtree_index, a.downcast()?, &self.py_builder_config)
            }
            PyStates::Value(PyValueStates::U16Equal(generic)) => {
                generic.build_subtree(py, subtree_index, a.downcast()?, &self.py_builder_config)
            }
            PyStates::Value(PyValueStates::U32Equal(generic)) => {
                generic.build_subtree(py, subtree_index, a.downcast()?, &self.py_builder_config)
            }
            PyStates::Value(PyValueStates::U64Equal(generic)) => {
                generic.build_subtree(py, subtree_index, a.downcast()?, &self.py_builder_config)
            }
            PyStates::Value(PyValueStates::F32Equal(generic)) => {
                generic.build_subtree(py, subtree_index, a.downcast()?, &self.py_builder_config)
            }
            PyStates::Value(PyValueStates::F64Equal(generic)) => {
                generic.build_subtree(py, subtree_index, a.downcast()?, &self.py_builder_config)
            }
            PyStates::Value(PyValueStates::I8SumThreshold(generic)) => {
                generic.build_subtree(py, subtree_index, a.downcast()?, &self.py_builder_config)
            }
        }
    }

    /// Builds the top tree from subtrees.
    ///
    /// It must be called after all subtrees are built with `build_subtree()`.
    ///
    /// Returns
    /// -------
    /// list of (int, numpy.ndarray of uint64, numpy.ndarray of float32/64)
    ///     List of (norder, indexes, values) tuples, a tuple per non-empty
    ///     norder.
    fn build_top_tree<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<(usize, &'py PyArray1<usize>, &'py PyUntypedArray)>> {
        let top_tree_config = self.py_builder_config.top_tree_config.clone();
        match &self.states {
            PyStates::MinMaxMean(PyMinMaxMeanStates::F32(generic)) => {
                generic.build_top_tree(py, top_tree_config)
            }
            PyStates::MinMaxMean(PyMinMaxMeanStates::F64(generic)) => {
                generic.build_top_tree(py, top_tree_config)
            }
            PyStates::Value(PyValueStates::BoolEqual(generic)) => {
                generic.build_top_tree(py, top_tree_config)
            }
            PyStates::Value(PyValueStates::I8Equal(generic)) => {
                generic.build_top_tree(py, top_tree_config)
            }
            PyStates::Value(PyValueStates::I16Equal(generic)) => {
                generic.build_top_tree(py, top_tree_config)
            }
            PyStates::Value(PyValueStates::I32Equal(generic)) => {
                generic.build_top_tree(py, top_tree_config)
            }
            PyStates::Value(PyValueStates::I64Equal(generic)) => {
                generic.build_top_tree(py, top_tree_config)
            }
            PyStates::Value(PyValueStates::U8Equal(generic)) => {
                generic.build_top_tree(py, top_tree_config)
            }
            PyStates::Value(PyValueStates::U16Equal(generic)) => {
                generic.build_top_tree(py, top_tree_config)
            }
            PyStates::Value(PyValueStates::U32Equal(generic)) => {
                generic.build_top_tree(py, top_tree_config)
            }
            PyStates::Value(PyValueStates::U64Equal(generic)) => {
                generic.build_top_tree(py, top_tree_config)
            }
            PyStates::Value(PyValueStates::F32Equal(generic)) => {
                generic.build_top_tree(py, top_tree_config)
            }
            PyStates::Value(PyValueStates::F64Equal(generic)) => {
                generic.build_top_tree(py, top_tree_config)
            }
            PyStates::Value(PyValueStates::I8SumThreshold(generic)) => {
                generic.build_top_tree(py, top_tree_config)
            }
        }
    }

    /// Extends from other MOMBuilder, clearing the other MOMBuilder.
    ///
    /// Parameters
    /// ----------
    /// other : MOMBuilder
    ///     MOMBuilder to extend from. It must have the same max_norder, split_norder, threshold,
    ///     and be built from the data of the same dtype.
    ///
    /// Returns
    /// -------
    /// None
    fn extend(&self, py: Python, other: &Self) -> PyResult<()> {
        if self.max_norder() != other.max_norder() {
            return Err(PyValueError::new_err("max_norder must be the same"));
        }
        if self.split_norder() != other.split_norder() {
            return Err(PyValueError::new_err("split_norder must be the same"));
        }
        if self.py_builder_config.mom_merger.dtype_char
            != other.py_builder_config.mom_merger.dtype_char
        {
            return Err(PyValueError::new_err("dtype must be the same"));
        }

        py.allow_threads(|| -> PyResult<()> {
            match (&self.states, &other.states) {
                (PyStates::MinMaxMean(PyMinMaxMeanStates::F32(self_generic)),
                 PyStates::MinMaxMean(PyMinMaxMeanStates::F32(other_generic))) => {
                    self_generic.extend(other_generic)
                }
                (PyStates::MinMaxMean(PyMinMaxMeanStates::F64(self_generic)),
                 PyStates::MinMaxMean(PyMinMaxMeanStates::F64(other_generic))) => {
                    self_generic.extend(other_generic)
                }
                (PyStates::Value(PyValueStates::BoolEqual(self_generic)),
                 PyStates::Value(PyValueStates::BoolEqual(other_generic))) => {
                    self_generic.extend(other_generic)
                }
                (PyStates::Value(PyValueStates::I8Equal(self_generic)),
                 PyStates::Value(PyValueStates::I8Equal(other_generic))) => {
                    self_generic.extend(other_generic)
                }
                (PyStates::Value(PyValueStates::I16Equal(self_generic)),
                 PyStates::Value(PyValueStates::I16Equal(other_generic))) => {
                    self_generic.extend(other_generic)
                }
                (PyStates::Value(PyValueStates::I32Equal(self_generic)),
                 PyStates::Value(PyValueStates::I32Equal(other_generic))) => {
                    self_generic.extend(other_generic)
                }
                (PyStates::Value(PyValueStates::I64Equal(self_generic)),
                 PyStates::Value(PyValueStates::I64Equal(other_generic))) => {
                    self_generic.extend(other_generic)
                }
                (PyStates::Value(PyValueStates::U8Equal(self_generic)),
                 PyStates::Value(PyValueStates::U8Equal(other_generic))) => {
                    self_generic.extend(other_generic)
                }
                (PyStates::Value(PyValueStates::U16Equal(self_generic)),
                 PyStates::Value(PyValueStates::U16Equal(other_generic))) => {
                    self_generic.extend(other_generic)
                }
                (PyStates::Value(PyValueStates::U32Equal(self_generic)),
                 PyStates::Value(PyValueStates::U32Equal(other_generic))) => {
                    self_generic.extend(other_generic)
                }
                (PyStates::Value(PyValueStates::U64Equal(self_generic)),
                 PyStates::Value(PyValueStates::U64Equal(other_generic))) => {
                    self_generic.extend(other_generic)
                }
                (PyStates::Value(PyValueStates::F32Equal(self_generic)),
                 PyStates::Value(PyValueStates::F32Equal(other_generic))) => {
                    self_generic.extend(other_generic)
                }
                (PyStates::Value(PyValueStates::F64Equal(self_generic)),
                 PyStates::Value(PyValueStates::F64Equal(other_generic))) => {
                    self_generic.extend(other_generic)
                }
                _ => {
                    panic!("Incompatible states, we should not reach this point because we have already checked the dtypes match")
                }
            }
        })
    }

    // pickle support
    fn __getnewargs_ex__(&self, py: Python<'_>) -> ((MomMerger,), HashMap<&'static str, PyObject>) {
        let merger = self.py_builder_config.mom_merger.clone();
        let max_norder = self.py_builder_config.max_norder.into_py(py);
        let split_norder = self.py_builder_config.split_norder.into_py(py);
        let thread_safe = self.py_builder_config.thread_safe.into_py(py);
        let args = (merger,);
        let kwargs = [
            ("max_norder", max_norder),
            ("split_norder", split_norder),
            ("thread_safe", thread_safe),
        ];
        (args, kwargs.into_iter().collect())
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
        let vec_bytes = serde_pickle::to_vec(&self, serde_pickle::SerOptions::new())
            .map_err(|err| PicklingError::new_err(format!("Cannot pickle MOMBuilder: {}", err)))?;
        Ok(PyBytes::new(py, &vec_bytes))
    }

    fn __setstate__(&mut self, state: &PyBytes) -> PyResult<()> {
        *self = serde_pickle::from_slice(state.as_bytes(), serde_pickle::DeOptions::new())
            .map_err(|err| {
                UnpicklingError::new_err(format!("Cannot unpickle MOMBuilder: {}", err))
            })?;
        Ok(())
    }

    // copy/deepcopy support

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: Py<PyAny>) -> Self {
        self.clone()
    }
}

impl<T, State, Merger> GenericStates<T, State, Merger>
where
    T: Into<State> + Copy + Sync + numpy::Element + 'static,
    State: Into<T> + Copy + Debug + Send + Sync,
    Merger: MergeStates<State = State> + Copy + Send + Sync,
{
    fn build_subtree<'py>(
        &self,
        py: Python<'py>,
        subtree_index: usize,
        a: &'py PyArray1<T>,
        config: &PyBuilderConfig,
    ) -> PyResult<Vec<(usize, &'py PyArray1<usize>, &'py PyUntypedArray)>> {
        py.allow_threads(|| {
            if self
                .states
                .read()
                .expect("Cannot lock states storage for read")
                .contains_key(&subtree_index)
            {
                Err(pyo3::exceptions::PyValueError::new_err(
                    "State with this index already exists",
                ))
            } else {
                Ok(())
            }
        })?;

        let py_ro = a.readonly();

        // We have to introduce an overhead here to avoid double locking of the states storage
        let tree = if config.thread_safe {
            let owned_array = py_ro.to_owned_array();
            py.allow_threads(|| self.build_subtree_impl(owned_array.view(), subtree_index, config))
        } else {
            let array_view = py_ro.as_array();
            self.build_subtree_impl(array_view, subtree_index, config)
        }?;

        // Return the rest of the subtree
        Ok(self.tree_to_python(py, tree, config.split_norder + 1))
    }

    fn build_subtree_impl(
        &self,
        array: NdArrayView<T>,
        subtree_index: usize,
        config: &PyBuilderConfig,
    ) -> PyResult<Tree<State>> {
        // We need to build a subtree with a single root node, we will accumulate all the nodes later
        // Current subtree has an offset in indexes on the deepest level (maximum norder)
        let index_offset = config.max_norder_index_offset(subtree_index);

        let it_states =
            array
                .iter()
                .enumerate()
                .map(|(relative_index, &x)| -> Result<_, Infallible> {
                    Ok((index_offset + relative_index, T::into(x)))
                });
        let mut tree = build_tree(self.merger, config.subtree_config.clone(), it_states)?;

        // Extract root node from the tree, it should have at most one state
        let root_tiles = tree.remove(0);
        let state = match root_tiles.len() {
            0 => None,
            1 => {
                let (indexes, states) = root_tiles.into_tuple();
                assert_eq!(indexes[0], subtree_index);
                Some(states[0])
            }
            _ => panic!("Root node should have at most one state"),
        };
        self.states
            .write()
            .expect("Cannot lock states storage for write")
            .insert(subtree_index, state);

        Ok(tree)
    }

    fn build_top_tree<'py>(
        &self,
        py: Python<'py>,
        top_tree_config: TreeConfig,
    ) -> PyResult<Vec<(usize, &'py PyArray1<usize>, &'py PyUntypedArray)>> {
        let tree = py.allow_threads(|| {
            {
                let states = self
                    .states
                    .read()
                    .expect("Cannot lock states storage for read");

                if states.len() != top_tree_config.max_norder_nleaves() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Not all subtrees are built",
                    ));
                }
                if states.keys().enumerate().any(|(i, index)| *index != i) {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Subtree tiles are not contiguous",
                    ));
                }
            }

            // Large memory allocation, we need to optimize it
            let states = std::mem::take(
                &mut *self
                    .states
                    .write()
                    .expect("Cannot lock states storage for write"),
            );

            let it_states = states.into_iter().filter_map(|(index, state)| {
                state.map(|state| -> Result<_, Infallible> { Ok((index, state)) })
            });

            Ok(build_tree(self.merger, top_tree_config, it_states)?)
        })?;

        Ok(self.tree_to_python(py, tree, 0))
    }

    fn tree_to_python<'py>(
        &self,
        py: Python<'py>,
        tree: Tree<State>,
        norder_offset: usize,
    ) -> Vec<(usize, &'py PyArray1<usize>, &'py PyUntypedArray)> {
        tree.into_iter()
            .enumerate()
            .filter(|(_norder, tiles)| tiles.len() != 0)
            .map(|(norder, tiles)| {
                let (indexes, values) = tiles.into_tuple();
                (
                    norder + norder_offset,
                    PyArray1::from_vec(py, indexes),
                    PyArray1::from_iter(py, values.into_iter().map(|state| state.into()))
                        .as_untyped(),
                )
            })
            .collect()
    }

    fn extend(&self, other: &Self) -> PyResult<()> {
        if other.subtree_states_is_empty() {
            // Nothing to merge with
            return Ok(());
        }

        let mut states = self
            .states
            .write()
            .expect("Cannot lock states storage for write");

        // We are cleaning the other's states storage here
        let other_states = std::mem::take(
            &mut *other
                .states
                .write()
                .expect("Cannot lock states storage for read"),
        );

        for (index, state) in other_states.into_iter() {
            if states.insert(index, state).is_some() {
                return Err(PyValueError::new_err(format!(
                    "State with index {index} already exists",
                )));
            }
        }

        Ok(())
    }

    fn subtree_states_is_empty(&self) -> bool {
        self.states
            .read()
            .expect("Cannot lock states storage for read")
            .is_empty()
    }
}

#[pymodule]
fn mom_builder(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MomMerger>()?;
    m.add_class::<MomBuilder>()?;
    Ok(())
}
