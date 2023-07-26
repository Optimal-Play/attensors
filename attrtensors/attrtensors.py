from typing import _AnnotatedAlias, Annotated, Any
from attrs import define, field, fields, asdict

import inspect
import numpy as np


class _AttrTensorMetaclass(type):
    mapped_methods = ['reshape', ]
    mapped_args = ['shape', 'dtype']

    def __getattr__(cls, key):
        def mapped_method(method, *args, **kwargs):
            for arg_name in kwargs:
                if arg_name not in cls.mapped_args:
                    raise TypeError(f"{cls.__name__} got an unexpected keyword argument {arg_name}")

            return cls(**{
                field.name: method(*args, **kwargs, 
                    shape=(kwargs, field.metadata.shape)) 
                for field in fields(cls)
            })

        callable = lambda method: bool(getattr(method, '__call__', None))

        method = getattr(np, key, None)
        
        if method is None or not callable(method):
            raise AttributeError(f"{cls.__name__} has no attribute {key}")
        
        return lambda *args, **kwargs: 
            cls(**{
                field.name: method(*args, **kwargs, 
                    shape=(kwargs, field.metadata.shape)) 
                for field in fields(cls)
            })

    @staticmethod
    def mapped_method(method):
        return lambda *args, **kwargs: \
            cls(**{
                field.name: method(*args, **kwargs, 
                    shape=(kwargs, field.metadata.shape)) 
                for field in fields(cls)
            })

    @staticmethod
    def callable(method):
        return bool(getattr(method, '__call__', None))

    @staticmethod
    def get_attribute(cls, key):
        pass


@define(slots=False)
class AttrTensor(np.ndarray):
    def __new__(cls, *args, shape=None, **kwargs):
        cls._tc_dtype = cls._dtype if hasattr(cls, "_dtype") else \
            AttrTensor.__infer_dtype__(cls)

        fieldnames = [field.name for field in fields(cls)]
        if set(kwargs.keys()) - set(fieldnames) != set():
            unexpected_arg = list(set(kwargs.keys()) - set(fieldnames))[0]
            raise TypeError(f"{cls.__name__} got an unexpected keyword argument '{unexpected_arg}'")

        n_args = len(args) + len(kwargs)
        n_required_args = len([f for f in fields(cls) if f.default is attrs.NOTHING])
        if n_args < n_required_args:
            raise TypeError(f"{cls.__name__} expected {n_required_args} arguments, got {n_args}")

        try: 
            mapped_inputs = AttrTensor.__map_inputs__(cls, *args, shape=shape, **kwargs)
        except:
            raise TypeError(f"Shape mismatch. Given shape should correspond to the shape prefix of each input array.")

        return np.asarray(mapped_inputs, dtype=cls._tc_dtype).view(cls)

    def __init__(self, *args, **kwargs):
        if all([getattr(self, field.name, None) is not None for field in fields(self.__class__)]):
            return
        self.__attrs_init__(**{
            field.name: self[field.name] for field in fields(self.__class__)
        })

    def __getitem__(self, key):
        ret = super().__getitem__(key)

        if ret.dtype != self.dtype:
            return ret.view(np.ndarray)

        if type(ret) == np.void:
            return np.asarray(ret, dtype=self.dtype).view(self.__class__)
            
        return ret

    def __array_finalize__(self, obj) -> None:
        if obj is None: return

        if self._tc_dtype != self.dtype:
            return

        self.__init__()

    @classmethod
    def broadcast(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> Any:
        # TODO: broadcast in and outside of dtype
        if all([type(i) == type(self) for i in inputs]):
            if ufunc.nout < 2 or method != "__call__":
                return self.__class__(**{
                    field.name: super(AttrTensor, self).__array_ufunc__(ufunc, method, *[getattr(i, field.name) for i in inputs], **kwargs)
                    for field in fields(self.__class__)
                })
            else:
                results = {
                    field.name: super(AttrTensor, self).__array_ufunc__(ufunc, method, *[getattr(i, field.name) for i in inputs], **kwargs)
                    for field in fields(self.__class__)
                }
                return (self.__class_(**{
                    }) for res in results)

        return super(AttrTensor, self).__array_ufunc__(ufunc, method, *inputs, **kwargs)

    @staticmethod
    def __infer_dtype__(cls) -> np.dtype:
        if not attrs.has(cls):
            raise TypeError(f"{cls.__name__} is not an attrs class (underlying dtype can't be inferred)")

        return np.dtype([
                (field.name, 
                field.metadata.get("dtype", None) or field.type
                if not issubclass(field.type, AttrTensor) else field.type.dtype,
                field.metadata.get("shape", ()))
                for field in fields(cls)
            ])

    @staticmethod
    def __map_inputs__(cls, *args, shape=None, **kwargs):            
        def tolist(arrays, shape):
            if len(shape) < 1:
                return tuple(arrays)
            return [tolist(tuple(a[i,...] for a in arrays), shape[1:]) for i in range(shape[0])]
        
        items = ()

        for field, arg in zip(fields(cls)[:len(args)], args):
            items = (*items, np.asarray(arg)) 

        for field in fields(cls)[len(args):]:
            items = (*items, np.asarray(kwargs[field.name]))

        return tolist(items, shape or ())


    @staticmethod
    def annotations_as_metadata(cls, fields):
        """Attrs field transformer function to convert annotated fields to metadata.
        Used with AttrTensor subclasses for easy definition."""
        annotated_fields = [
            f.evolve(type=cls.__annotations__[f.name].__origin__,
                metadata=cls.__annotations__[f.name].__metadata__[0])
            if f.name in cls.__annotations__ and
                issubclass(type(cls.__annotations__[f.name]), _AnnotatedAlias) else f
            for f in fields
        ]

        assert all([
            'dtype' in f.metadata and 'shape' in f.metadata
            for f in annotated_fields
            if f.name in cls.__annotations__ and
                issubclass(type(cls.__annotations__[f.name]), _AnnotatedAlias)
        ]), "All annotated fields must have dtype and shape metadata"

        return annotated_fields

    @staticmethod
    def define(maybe_cls=None, **kwargs):
        if 'slots' in kwargs or 'init' in kwargs:
            raise TypeError("Slots and init are forced to False for AttrTensor subclasses.")

        def subclass_tc(cls):
            if issubclass(cls, AttrTensor):
                return cls
            return type(cls.__name__, (AttrTensor, ) + cls.__bases__, dict(cls.__dict__))

        chained_field_transformer = \
            AttrTensor.annotations_as_metadata if "field_transformer" not in kwargs else \
            lambda cls, fields: kwargs["field_transformer"](cls, AttrTensor.annotations_as_metadata(cls, fields)) 

        def decorate(maybe_cls):
            new_class = define(subclass_tc(maybe_cls), **kwargs, 
                init=False, slots=False, field_transformer=chained_field_transformer)
            
            new_class._tc_dtype = AttrTensor.__infer_dtype__(new_class)

            return new_class

        if not maybe_cls:
            return lambda maybe_cls: decorate(maybe_cls)

        return decorate(maybe_cls)