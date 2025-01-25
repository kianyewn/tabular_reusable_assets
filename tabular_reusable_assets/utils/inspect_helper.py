import inspect
from functools import partial
from typing import Callable


def is_var_positional_or_var_keyword(param: inspect.Parameter):
    """Check if parameter is var positional or var keyword.

    Args:
        param (inspect.Parameter): Parameter to check.

    Returns:
        bool: True if parameter is var positional or var keyword, False otherwise.
    """
    return param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def is_empty(param: inspect.Parameter):
    """Check if parameter is empty.

    Args:
        param (inspect.Parameter): Parameter to check.

    Returns:
        bool: True if parameter is empty, False otherwise.
    """
    return param.default is param.empty


def is_keyword_only(param: inspect.Parameter):
    """Check if parameter is keyword only.

    Args:
        param (inspect.Parameter): Parameter to check.

    Returns:
        bool: True if parameter is keyword only, False otherwise.
    """
    return param.kind == param.KEYWORD_ONLY


def get_all_parameters_name(func: Callable):
    """Get all parameters name.

    Args:
        func (inspect.Parameter): Function to get parameters name.

    Returns:
        list: List of parameters name.
    """
    return list(inspect.signature(func).parameters.keys())


def get_all_parameters_name_and_annotation(func: Callable):
    """Get all parameters name and annotation.

    Args:
        func (Callable): Function to get parameters name and annotation.

    Returns:
        dict: Dictionary of parameters name and annotation.
    """
    return {name: param.annotation for name, param in inspect.signature(func).parameters.items()}


def get_all_parameters_kind(func: Callable):
    """Get all parameters kind.

    Args:
        func (Callable): Function to get parameters kind.

    Returns:
        dict: Dictionary of parameters kind.
    """
    return {name: p.kind for name, p in inspect.signature(func).parameters.items()}


def get_non_empty_parameters(func: Callable):
    """Get non empty parameters.

    Args:
        func (Callable): Function to get non empty parameters.

    Returns:
        list: List of non empty parameters.
    """
    return [
        name
        for name, param in inspect.signature(func).parameters.items()
        if not is_empty(param) and not is_var_positional_or_var_keyword(param)
    ]


def get_empty_parameters(func: Callable):
    """Get empty parameters.

    Args:
        func (Callable): Function to get empty parameters.

    Returns:
        list: List of empty parameters.
    """
    return [
        name
        for name, param in inspect.signature(func).parameters.items()
        if is_empty(param) and not is_var_positional_or_var_keyword(param)
    ]


def get_required_parameters(func: Callable):
    """Get required parameters.

    Args:
        func (Callable): Function to get required parameters.

    Returns:
        list: List of required parameters.
    """
    return get_empty_parameters(func)


def get_keyword_only_parameters(func: Callable):
    """Get keyword only parameters.

    Args:
        func (Callable): Function to get keyword only parameters.

    Returns:
        list: List of keyword only parameters.
    """
    return [name for name, param in inspect.signature(func).parameters.items() if param.kind == param.KEYWORD_ONLY]


def get_empty_keyword_only_parameters(func: Callable):
    """Get empty keyword only parameters.

    Args:
        func (Callable): Function to get empty keyword only parameters.

    Returns:
        list: List of empty keyword only parameters.
    """
    return [
        name
        for name, param in inspect.signature(func).parameters.items()
        if is_keyword_only(param) and is_empty(param)
    ]


def get_var_keyword_name(func: Callable):
    """Get var keyword name.

    Args:
        func (Callable): Function to get var keyword name.

    Returns:
        list: List of var keyword name.
    """
    return [name for name, param in inspect.signature(func).parameters.items() if param.kind == param.VAR_KEYWORD]


def get_var_positional_name(func: Callable):
    """Get var positional name.

    Args:
        func (Callable): Function to get var positional name.

    Returns:
        list: List of var positional name.
    """
    return [name for name, param in inspect.signature(func).parameters.items() if param.kind == param.VAR_POSITIONAL]


def get_partial_function(partial_func: partial):
    """Get non empty arguments and all parameters from a partial function.

    Args:
        partial_func (partial): Partial function to get non empty arguments and all parameters.

    Returns:
        tuple: Tuple of non empty arguments and all parameters.
    """
    # p = partial(lambda a, b: a > b, {"a": 1})
    non_empty_args = partial_func.args  # ({'a': 1},)
    all_parameters = partial_func.func.__code__.co_varnames  # ('a', 'b')
    return non_empty_args, all_parameters


if __name__ == "__main__":

    def func(a, b, c, *args, w, d=2, **kwargs):
        return locals()

    # check parameter type understanding
    assert inspect.signature(func).parameters["a"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert inspect.signature(func).parameters["b"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert inspect.signature(func).parameters["c"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert inspect.signature(func).parameters["args"].kind == inspect.Parameter.VAR_POSITIONAL
    assert inspect.signature(func).parameters["w"].kind == inspect.Parameter.KEYWORD_ONLY
    assert inspect.signature(func).parameters["d"].kind == inspect.Parameter.KEYWORD_ONLY
    assert inspect.signature(func).parameters["kwargs"].kind == inspect.Parameter.VAR_KEYWORD

    assert (get_all_parameters_name(func)) == ["a", "b", "c", "args", "w", "d", "kwargs"]

    assert (get_non_empty_parameters(func)) == ["d"]
    assert get_empty_parameters(func) == ["a", "b", "c", "w"]
    # print(get_all_parameters_name_and_annotation(func))
    assert (get_keyword_only_parameters(func)) == ["w", "d"]
    assert (get_empty_keyword_only_parameters(func)) == ["w"]

    assert (get_required_parameters(func)) == ["a", "b", "c", "w"]

    assert get_var_keyword_name(func) == ["kwargs"]
    assert get_var_positional_name(func) == ["args"]

    def func(
        a,
        b,
        c,
        w,
        d=2,
    ):
        return locals()

    # check parameter type understanding
    # notice how after removing *args, the parameters are now positional or keyword
    assert inspect.signature(func).parameters["a"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert inspect.signature(func).parameters["b"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert inspect.signature(func).parameters["c"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert inspect.signature(func).parameters["w"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert inspect.signature(func).parameters["d"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
