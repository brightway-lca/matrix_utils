def get_exactly_one(iterator):
    objs = list(iterator)
    if len(objs) != 1:
        raise ValueError("Multiple values found where only one is allowed")
    return objs[0]
