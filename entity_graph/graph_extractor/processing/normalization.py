from enum import StrEnum


class NormalizationMethod(StrEnum):
    NONE: str = "none"
    ONE_SEPERATOR: str = "one_separator"


none_normalization = lambda x: x
separator_normalization = lambda x: x


def get_normalization_method(normalization_method: NormalizationMethod):
    """Returns normalization function for selected method."""
    if normalization_method == NormalizationMethod.NONE:
        return none_normalization
    if normalization_method == NormalizationMethod.ONE_SEPERATOR:
        return separator_normalization
