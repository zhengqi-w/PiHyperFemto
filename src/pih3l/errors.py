"""Errors that should be actionable at the command line."""


class AnalysisError(Exception):
    """Invalid input, configuration, or analysis state."""


class SchemaError(AnalysisError):
    """An input does not satisfy the declared data contract."""
