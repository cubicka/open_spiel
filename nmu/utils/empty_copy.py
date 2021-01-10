def empty_copy(obj):
    class Empty(obj.__class__):
        def __init__(self): pass
    newcopy = Empty()
    newcopy.__class__ = obj.__class__
    return newcopy

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
