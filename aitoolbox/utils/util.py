import types
import functools


def function_exists(object_to_check, fn_name):
    """Check if function exists in the object

    Args:
        object_to_check: object to be searched for the existence of the function
        fn_name (str): name of the function

    Returns:
        bool: if function is present in the provided object
    """
    if hasattr(object_to_check, fn_name):
        fn_obj = getattr(object_to_check, fn_name, None)
        return callable(fn_obj)
    return False


def copy_function(fn):
    """Deep copy a function

    Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)

    Args:
        fn (callable): original function

    Returns:
        callable: copy of the provided function
    """
    g = types.FunctionType(fn.__code__, fn.__globals__, name=fn.__name__,
                           argdefs=fn.__defaults__,
                           closure=fn.__closure__)
    g = functools.update_wrapper(g, fn)
    g.__kwdefaults__ = fn.__kwdefaults__
    return g


def is_empty_function(fn):
    """Returns true if f is an empty function

    Taken from StackOverflow:
        https://stackoverflow.com/a/58973125

    Args:
        fn: function to be evaluated if it is empty or not

    Returns:
        bool: true if provided function is empty, otherwise false
    """
    def empty_func():
        pass

    def empty_func_with_docstring():
        """Empty function with docstring."""
        pass

    empty_lambda = lambda: None
    empty_lambda_with_docstring = lambda: None
    empty_lambda_with_docstring.__doc__ = """Empty function with docstring."""

    def constants(f):
        """Return a tuple containing all the constants of a function without:
            * docstring
        """
        return tuple(x for x in f.__code__.co_consts if x != f.__doc__)

    return (
                   fn.__code__.co_code == empty_func.__code__.co_code and
                   constants(fn) == constants(empty_func)
           ) or (
                   fn.__code__.co_code == empty_func_with_docstring.__code__.co_code and
                   constants(fn) == constants(empty_func_with_docstring)
           ) or (
                   fn.__code__.co_code == empty_lambda.__code__.co_code and
                   constants(fn) == constants(empty_lambda)
           ) or (
                   fn.__code__.co_code == empty_lambda_with_docstring.__code__.co_code and
                   constants(fn) == constants(empty_lambda_with_docstring)
           )


def flatten_list_of_lists(nested_list):
    """Flatten the nested list of lists

    Args:
        nested_list (list): nested list of lists to be flattened

    Returns:
        list or None: flattened list
    """
    if nested_list is not None:
        return [item for sublist in nested_list for item in sublist]
    else:
        return None
