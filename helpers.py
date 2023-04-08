def process_attr(func):
    def _inner(obj,val):
        res = getattr(obj,val)
        return func(val,res)
    return _inner


