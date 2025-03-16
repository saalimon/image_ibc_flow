from functools import wraps


def logger(fn):
    '''
    Decorator to log function name and arguments
    param
        fn: function, function to be decorated
    return
        function, decorated function
    '''
    from datetime import datetime, timezone

    @wraps(fn)
    def inner(*args, **kwargs):
        called_at = datetime.now(timezone.utc)
        print(f">>> Running {fn.__name__!r} function. Logged at {called_at}")
        to_execute = fn(*args, **kwargs)
        print(f">>> Function: {fn.__name__!r} executed. Logged at {called_at}")
        return to_execute

    return inner
