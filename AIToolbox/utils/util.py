
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


def flatten_list_of_lists(l):
    """

    Args:
        l (list):

    Returns:
        list or None:
    """
    if l is not None:
        return [item for sublist in l for item in sublist]
    else:
        return None
