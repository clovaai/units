from dataclasses import fields


class InstanceType:
    COORDS = 1
    MASKS = 2


class Fields:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def set(self, key, value):
        self.__dict__[key] = value

    def fields(self):
        return self.__dict__

    def to(self, *args, **kwargs):
        kv = self.__dict__
        res = {}

        for k, v in kv.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)

            elif isinstance(v, list):
                res_list = []

                for e in v:
                    if hasattr(e, "to"):
                        res_list.append(e.to(*args, **kwargs))

                    else:
                        res_list.append(e)

                v = res_list

            res[k] = v

        return self.__class__(**res)


class Sample(Fields):
    def __init__(self, image_size, **kwargs):
        super().__init__(image_size=image_size, **kwargs)


class Batch(Fields):
    pass


class FieldsMixin:
    def to(self, *args, **kwargs):
        keys = fields(self)
        res = {}

        for k in keys:
            k = k.name
            v = getattr(self, k)

            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)

            elif isinstance(v, list):
                res_list = []

                for e in v:
                    if hasattr(e, "to"):
                        res_list.append(e.to(*args, **kwargs))

                    else:
                        res_list.append(e)

                v = res_list

            res[k] = v

        return self.__class__(**res)
