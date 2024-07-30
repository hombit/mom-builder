/// Trait for checking if the merged state is valid.
pub trait StateIsValid {
    /// Type of the leaf state.
    type State;

    /// Checks if the given state is valid, i.e. if it should be stored in the tree.
    fn state_is_valid(&self, state: &Self::State) -> bool;
}
