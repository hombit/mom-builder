use crate::state::merge_states::MergeStates;
use crate::state::state_is_valid::StateIsValid;
use numpy::ndarray::NdFloat;
use serde::{Deserialize, Serialize};

/// Leaf state with minimum, maximum and mean values.
///
/// It implements [Into] trait for [f32] and [f64], so it can be converted to its mean value.
/// [From] trait for `T` which just assigns the given value to all three fields.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MinMaxMeanState<T> {
    /// Minimum value.
    pub min: T,
    /// Maximum value.
    pub max: T,
    /// Mean value.
    pub mean: T,
}

impl<T> MinMaxMeanState<T>
where
    T: Copy,
{
    /// Creates a new [MinMaxMeanState] with the given value, which is used as minimum, maximum and
    /// mean.
    pub fn new(value: T) -> Self {
        Self {
            min: value,
            max: value,
            mean: value,
        }
    }
}

impl From<MinMaxMeanState<f32>> for f32 {
    fn from(val: MinMaxMeanState<f32>) -> Self {
        val.mean
    }
}

impl From<MinMaxMeanState<f64>> for f64 {
    fn from(val: MinMaxMeanState<f64>) -> Self {
        val.mean
    }
}

impl<T> From<T> for MinMaxMeanState<T>
where
    T: Copy,
{
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

/// Merges leaf states by taking minimum and maximum of the states and calculating the mean value.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Merger<V> {
    pub validator: V,
}

impl<V> Merger<V> {
    pub fn new(validator: V) -> Self {
        Self { validator }
    }
}

impl<T, V> MergeStates for Merger<V>
where
    T: NdFloat,
    V: StateIsValid<State = MinMaxMeanState<T>>,
{
    type State = MinMaxMeanState<T>;

    /// Merges the given states by taking minimum and maximum of the states and calculating the mean
    /// value.
    fn merge(&self, states: &[Self::State]) -> Option<Self::State> {
        assert!(!states.is_empty());

        let mut min = states[0].min;
        let mut max = states[0].max;
        let mut sum = states[0].mean;
        for state in states.iter().skip(1) {
            if state.min < min {
                min = state.min;
            }
            if state.max > max {
                max = state.max;
            }
            sum += state.mean;
        }
        let merged_state = Self::State {
            min,
            max,
            mean: sum / T::from(states.len()).expect("N cannot be casted to float"),
        };

        if self.validator.state_is_valid(&merged_state) {
            Some(merged_state)
        } else {
            None
        }
    }
}

/// Checks if the relative difference between minimum and maximum is less than a given threshold.
///
/// Basically it checks if `abs(max - min) / max <= threshold`, but with some edge-case handling.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct RelativeToleranceValidator<T> {
    threshold: T,
}

impl<T> RelativeToleranceValidator<T>
where
    T: NdFloat,
{
    /// Creates a new [RelativeToleranceValidator] with the given threshold.
    ///
    /// The threshold must be non-negative, otherwise the method panics.
    pub fn new(threshold: T) -> Self {
        assert!(threshold >= T::zero());
        Self { threshold }
    }

    /// Returns the threshold.
    pub fn threshold(&self) -> T {
        self.threshold
    }
}

impl<T> StateIsValid for RelativeToleranceValidator<T>
where
    T: NdFloat,
{
    type State = MinMaxMeanState<T>;

    /// Checks if the relative difference between minimum and maximum is less than the threshold.
    ///
    /// Basically it checks if `abs(max - min) / norm <= threshold`, where `norm` is the maximum of
    /// the absolute values of `min` and `max`. If both are zero, the method returns `true`.
    fn state_is_valid(&self, state: &Self::State) -> bool {
        let denominator = T::max(T::abs(state.min), T::abs(state.max));
        if denominator.is_zero() {
            return true;
        }
        let ratio = (state.max - state.min) / denominator;
        ratio <= self.threshold
    }
}
