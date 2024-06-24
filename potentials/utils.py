"""

[1] https://code.activestate.com/recipes/577504/
"""
import sys
import dataclasses
import itertools

import numpy as np

from potentials import cluster

def _getsizeof(o: object, default_size=8):
    """Avoid taking into account garbage collector overhead """
    try:
        return o.__sizeof__()
    except AttributeError as ar:
        return default_size


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}
    """
    dict_handler = lambda d: itertools.chain.from_iterable(d.items())
    all_handlers = {
        tuple: iter,
        list: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = sys.getsizeof(
        0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = _getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o))

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break  # Avoid multiple conatiner counts

        else:
            if not hasattr(
                    o.__class__,
                    '__slots__'):  # no __slots__ *usually* means a __dict__,
                if hasattr(
                        o, '__dict__'
                ):  # some special builtin classes (such as `type(None)`) have neither
                    s += sizeof(o.__dict__)
            else:  # else, `o` has no attributes at all, so _getsizeof() actually returned the correct value
                s += sum(
                    sizeof(getattr(o, x)) for x in o.__class__.__slots__
                    if hasattr(o, x))
        return s

    return sizeof(o)


