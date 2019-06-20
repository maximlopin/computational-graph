import time
import inspect

history = {}

def track(f):
    def wrapper(*args):

        start = time.time()
        ret = f(*args)
        end = time.time()
        diff = (end - start)

        if f.__qualname__ not in history:
            history[f.__qualname__] = 0.0
        else:
            history[f.__qualname__] += diff

        return ret
    return wrapper

def summarize():

    global history
    total_time = sum([t for f, t in history.items()])

    history = dict(sorted(history.items(), key=lambda kv: kv[1], reverse=True))

    print('Total time:', total_time)

    for f, t in history.items():
        print('{:.2f} ms ({:.2f} %)'.format(1000*t, 100.0 * t / total_time), f)

def for_all_methods(decorator):
    def decorate(cls):
        for name, fn in inspect.getmembers(cls):
            if isinstance(fn, inspect.types.FunctionType):
                setattr(cls, name, decorator(fn))
        return cls
    return decorate
