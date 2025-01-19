import inspect

import tabular_reusable_assets.utils.inspect_helper as inspect_helper


def test_inspect_helper():
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

    assert (inspect_helper.get_all_parameters_name(func)) == ["a", "b", "c", "args", "w", "d", "kwargs"]

    assert (inspect_helper.get_non_empty_parameters(func)) == ["d"]
    assert (inspect_helper.get_empty_parameters(func)) == ["a", "b", "c", "w"]
    # print(get_all_parameters_name_and_annotation(func))
    assert (inspect_helper.get_keyword_only_parameters(func)) == ["w", "d"]
    assert (inspect_helper.get_empty_keyword_only_parameters(func)) == ["w"]

    assert (inspect_helper.get_required_parameters(func)) == ["a", "b", "c", "w"]

    assert (inspect_helper.get_var_keyword_name(func)) == ["kwargs"]
    assert (inspect_helper.get_var_positional_name(func)) == ["args"]

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
