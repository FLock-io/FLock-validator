"""Failure categories at the boundary between a submitter and a validator."""


class InvalidSubmission(ValueError):
    """The submitted artifact does not meet the model contract."""


class ModelParameterLimit(InvalidSubmission):
    """The model exceeds the assignment's parameter budget."""


# Same penalty as validator/modules/rl in the supplied FLock reference.
LOWEST_POSSIBLE_REWARD = -999.0
