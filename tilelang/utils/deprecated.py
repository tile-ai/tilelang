def deprecated_warning(method_name: str, new_method_name: str, phaseout_version: str = None):
    """A function to indicate that a method is deprecated"""
    import warnings  # pylint: disable=import-outside-toplevel, import-error

    warnings.warn(
        f"{method_name} is deprecated, use {new_method_name} instead"
        + (f" and will be removed in {phaseout_version}" if phaseout_version else ""),
        DeprecationWarning,
        stacklevel=2,
    )


def deprecated(
    method_name: str,
    new_method_name: str,
    phaseout_version: str = None,
):
    """A decorator to indicate that a method is deprecated

    Parameters
    ----------
    method_name : str
        The name of the method to deprecate
    new_method_name : str
        The name of the new method to use instead
    phaseout_version : str
        The version to phase out the method
    """
    import functools  # pylint: disable=import-outside-toplevel

    def _deprecate(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            deprecated_warning(method_name, new_method_name, phaseout_version)
            return func(*args, **kwargs)

        return _wrapper

    return _deprecate
