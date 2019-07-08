
def function_exists(object_to_check, fn_name):
    """

    Args:
        object_to_check:
        fn_name (str):

    Returns:
        bool:
    """
    if hasattr(object_to_check, fn_name):
        fn_obj = getattr(object_to_check, fn_name, None)
        return callable(fn_obj)
    return False
