use crate::state::merge_states::MergeStates;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Simple leaf state with a single value.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ValueState<T>(pub T);

macro_rules! impl_from_value_state {
    ($($t:ty),*) => {
        $(
            impl From<ValueState<$t>> for $t {
                fn from(state: ValueState<$t>) -> Self {
                    state.0
                }
            }
        )*
    };
    () => {};
}

impl_from_value_state!(bool, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, f32, f64, String);

impl<T> From<T> for ValueState<T> {
    fn from(val: T) -> Self {
        Self(val)
    }
}

/// Merges leaf states if values are exactly equal.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct ExactlyEqualMerger<T> {
    phantom: PhantomData<T>,
}

impl<T> ExactlyEqualMerger<T> {
    fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<T> Default for ExactlyEqualMerger<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> MergeStates for ExactlyEqualMerger<T>
where
    T: PartialEq + Copy,
{
    type State = ValueState<T>;

    fn merge(&self, states: &[Self::State]) -> Option<Self::State> {
        assert!(!states.is_empty());

        // Output any state if all states are equal, otherwise return None
        let first_state = states[0];
        if states.iter().skip(1).all(|&state| state == first_state) {
            Some(first_state)
        } else {
            None
        }
    }
}
