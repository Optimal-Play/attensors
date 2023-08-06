from functools import partial
from typing import _AnnotatedAlias, Annotated, Any
from attrs import define, field, fields

import attrs
import numpy as np


def on_setattr(instance, attrib, new_value):
    if not getattr(instance, attrib.name, None):
        return setattr(instance, attrib.name, new_value)
    instance[attrib.name] = new_value


class _TensorsMetaclass(type):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mapped_methods = [
            # array creation routines
            'empty', 'empty_like', 'ones', 'ones_like',
            'zeros', 'zeros_like', 'full', 'full_like',
            # array manipulation routines
            'concatenate', 'stack',
            'reshape', 'ravel',
            # padding
            'pad'
        ]

        for method in mapped_methods:
            setattr(cls, method, cls.get_implementation(method))

    def get_implementation(cls, method):
        def mapped_method(method, *args, **kwargs):
            if cls is not Tensors:
                if 'dtype' in kwargs:
                    raise TypeError(f"{cls.__name__}.{method.__name__} " + \
                        "got an unexpected keyword argument 'dtype'")
                kwargs['dtype'] = cls._dtype

            return method(*args, **kwargs).view(cls)

        return partial(mapped_method, getattr(np, method))


@define(slots=False, on_setattr=on_setattr)
class Tensors(np.ndarray, metaclass=_TensorsMetaclass):
    """Extension of np.ndarray.
    Essentially, a structured array with defined rules for universal functions and
    some shortcuts to common numpy functions, providing already known arguments (e.g. dtype, subok).

    Subclasses of Tensors are expected to be attrs type classes. Subclassing is done by using
    the `@tensors` decorator. Alternatively, attrs' `@define` decorator can also be used, but
    it would be required to add specific configurations to function as intended - which `@tensors`
    already provides.

    Directly instantiating Tensors is possible. In this case, you provide the fields and values
    you want as keyword arguments. Either `shape` or `dtype` must be provided. Providing one
    will cause the other to be inferred.
    If only `shape` is provided, this will be considered the shape of the instance, which must
    prefix all included tensors shapes. The `dtype` will be inferred based on the provided values.
    If only `dtype` is provided, the instance's `shape` will be considered () as long as `mc_shape`
    is False. If `mc_shape` is set to True, the instance's `shape` will be computed as the
    maximum common shape amongst the provided values, while respecting the provided `dtype`.
    Providing both is exhaustive and will cause validation of the provided `shape` as `dtype`
    takes precedence.

    Arguments `shape`, `dtype`, `mc_shape` are NOT available for subclasses of Tensors.
    Instead, the dtype is inferred from the class fields information (type and metadata).

    Args:
        *args: Positional input tensors
        **kwargs: Other keyword input tensors
        shape (Tuple[int]): Shape of the tensor collection
        dtype (np.dtype): Data type of the tensor collection
        mc_shape (bool): Only takes effect if neither shape nor dtype are specified.
            The shape will be computed as the maximum common shape amongst input tensors,
            otherwise it is set to ()
    """
    def __new__(cls, *args, shape=None, dtype=None, mc_shape=None, **kwargs):
        for key in kwargs.keys():
            if hasattr(np.ndarray, key):
                raise TypeError(f"`{key}` is a np.ndarray reserved attribute and cannot be used.")

        inputs = {
            **({f'f{i}': np.asarray(arg) for i, arg in enumerate(args)} if not fields(cls) else
                {f.name: np.asarray(arg) for f, arg in zip(fields(cls), args)}),
            **{name: np.asarray(value) for name, value in kwargs.items()}
        }

        if cls is Tensors:
            inputs, cls._dtype, shape = cls.__base_constructor__(
                inputs, shape=shape, dtype=dtype, mc_shape=mc_shape)
        else:
            inputs, cls._dtype, shape = cls.__subclass_constructor__(
                inputs, shape=shape, dtype=dtype, mc_shape=mc_shape)

        inputs = {k: inputs[k] for k in cls._dtype.names}

        mapped_inputs = cls.__map_inputs__(inputs, shape)

        return np.asarray(mapped_inputs, dtype=cls._dtype).view(cls)

    def __init__(self, *args, **kwargs):
        if all([getattr(self, field.name, None) is not None for field in fields(self.__class__)]):
            return

        self.__attrs_init__(**{
            field.name: self[field.name] for field in fields(self.__class__)
        })

    def __array_finalize__(self, obj) -> None:
        if obj is None: return

        if type(self) == Tensors:
            return

        if self._dtype != self.dtype:
            return

        self.__init__()

    def __getitem__(self, key):
        ret = super().__getitem__(key)

        if ret.dtype.names is None:
            return ret.view(np.ndarray)

        if type(self) is not Tensors and type(key) is str:
            if issubclass(getattr(fields(self.__class__), key).type, Tensors):
                return ret.view(getattr(fields(self.__class__), key).type)

        # Automatically downcast to Tensors if dtypes don't match
        if ret.dtype != self.dtype:
            return ret.view(Tensors)

        if type(ret) == np.void:
            return np.asarray(ret).view(self.__class__)

        return ret

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}" + "([" + \
            ", ".join([f"{k}={repr(self[k])}" for k in self.dtype.names]) + "])"

    def asdict(self):
        if not hasattr(self, '_dict'):
            self._dict = {k: self[k] for k in self.dtype.names}
        return self._dict

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> Any:
        broadcasted = self.broadcast_tensors(*inputs)

        if ufunc.nout < 2 or method != "__call__":
            return self.__class__(**{
                field_name: super(Tensors, self).__array_ufunc__(
                    ufunc, method, *[i[field_name] for i in broadcasted], **kwargs)
                for field_name in self.dtype.names
            }, shape=(broadcasted[0].shape if type(self) == Tensors else None))
        elif ufunc.nout > 1:
            results = {
                field_name: super(Tensors, self).__array_ufunc__(
                    ufunc, method, *[i[field_name]for i in broadcasted], **kwargs)
                for field_name in self.dtype.names
            }
            return (self.__class_(**{
                field_name: results[field_name][i]
                for field_name in self.dtype.names
            }, shape=(broadcasted[0].shape if type(self) == Tensors else None))
            for i in range(len(list(results.values)[0])))

        return super(Tensors, self).__array_ufunc__(ufunc, method, *inputs, **kwargs)

    @classmethod
    def __base_constructor__(cls, inputs, shape, dtype, mc_shape):
        if len(inputs) == 0:
                raise TypeError("Tensors() takes at least one argument (0 given)")

        if dtype:
            if not isinstance(dtype, np.dtype):
                dtype = np.dtype(dtype)
            if not dtype.fields:
                raise TypeError(f"Invalid or non-structured dtype {dtype}")

        def common_shape(inputs):
            shape = ()
            for dims in zip(*[i.shape for i in inputs.values()]):
                dims = set(dims)
                if len(dims) > 1:
                    return shape
                shape = (*shape, list(dims)[0])
            return shape

        def external_shape(inputs, dtype):
            first = list(inputs.values())[0]
            first_dt_shape = list(dtype.fields.values())[0][0].shape
            split = -len(first_dt_shape) if len(first_dt_shape) else len(first.shape)
            return first.shape[:split]

        try:
            shape = shape or (external_shape(inputs, dtype) if dtype else None)
            shape = shape or (common_shape(inputs) if mc_shape else ())
            dtype = dtype or np.dtype([
                (name, value.dtype, value.shape[len(shape):])
                for name, value in inputs.items()
            ])
        except:
            raise TypeError(f"Shape mismatch. " + \
                "Given shape should correspond to the shape prefix of each input array.")

        if any([v.shape[:len(shape)] != shape or v.shape[len(shape):] != dtype[fn].shape
                for fn, v in inputs.items()]):
            raise TypeError(f"Shape mismatch. " + \
                "Given shape should correspond to the shape prefix of each input array.")

        return inputs, dtype, shape

    @classmethod
    def __subclass_constructor__(cls, inputs, shape, dtype, mc_shape):
        cls_dtype = cls._dtype if hasattr(cls, "_dtype") else cls.__infer_dtype__()

        fieldnames = [field.name for field in fields(cls)]
        kwds = list(inputs.keys()) + \
            (['shape'] if shape is not None else []) + \
            (['dtype'] if dtype is not None else []) + \
            (['mc_shape'] if mc_shape is not None else [])
        if set(kwds) - set(fieldnames) != set():
            unexpected_arg = list(set(kwds) - set(fieldnames))[0]
            raise TypeError(f"__init__() got an unexpected keyword argument '{unexpected_arg}'")

        n_args = len(inputs)
        n_required_args = len([f for f in fields(cls) if f.default is attrs.NOTHING])
        if n_args < n_required_args:
            raise TypeError(f"{cls.__name__} expected {n_required_args} arguments, got {n_args}")

        ext_shapes = []
        for f, v in inputs.items():
            split = len(v.shape) if not len(cls_dtype[f].shape) else -len(cls_dtype[f].shape)
            if v.shape[split:] != cls_dtype[f].shape:
                raise TypeError(f"Shape mismatch. " + \
                    "Given shape should correspond to the shape prefix of each input array.")
            ext_shapes.append(v.shape[:split])

        ext_shapes = set(ext_shapes)
        if len(ext_shapes) > 1:
            raise TypeError(f"Shape mismatch. " + \
                "Given shape should correspond to the shape prefix of each input array.")

        return inputs, cls_dtype, list(ext_shapes)[0]

    @classmethod
    def __infer_dtype__(cls) -> np.dtype:
        if not attrs.has(cls):
            raise TypeError(f"{cls.__name__} " + \
                "is not an attrs class (underlying dtype cannot be inferred)")

        return np.dtype([
                (field.name,
                field.metadata.get("dtype", None) or field.type
                if not issubclass(field.type, Tensors) else field.type._dtype,
                field.metadata.get("shape", ()))
                for field in fields(cls)
            ])

    @classmethod
    def __map_inputs__(cls, inputs, shape):
        def tolist(arrays, shape):
            if len(shape) < 1:
                return tuple(arrays)
            return [tolist(tuple(a[i,...] for a in arrays), shape[1:]) for i in range(shape[0])]

        return tolist(tuple(inputs.values()), shape)

    @classmethod
    def broadcast_tensors(cls, *args):
        tensors = [arg for arg in args if isinstance(arg, Tensors)]
        if not len(tensors):
            raise ValueError("No tensors found in arguments.")
        init_dtype = tensors[0].dtype
        fields = set(arg.dtype.names for arg in tensors)
        if len(fields) > 1:
            raise ValueError("Cannot broadcast tensors with different fields.")

        field_names = list(fields)[0]
        field_shapes = [
            np.broadcast_shapes(*[arg.dtype[fn].shape for arg in tensors])
            for fn in field_names
        ]
        outer_shape = np.broadcast_shapes(*[arg.shape for arg in tensors])

        broadcasted = []
        for arg in args:
            if not isinstance(arg, Tensors):
                arg = np.asarray(arg)
                arg = Tensors(**{
                    fn: np.broadcast_to(arg, (*outer_shape, *fshape))
                    for fn, fshape in zip(field_names, field_shapes)
                }, shape=outer_shape).view(cls)
            elif arg.dtype == init_dtype:
                arg = np.broadcast_to(arg, outer_shape, subok=True)
            else:
                arg_fields = {}

                for fn, fshape in zip(field_names, field_shapes):
                    outer_slices = (slice(None),) * len(arg.shape)
                    expand_dims = (None,) * (len(fshape) - len(arg.dtype[fn].shape))
                    arg_fields[fn] = arg[fn][(*outer_slices, *expand_dims, ...)]

                arg = Tensors(**{
                    fn: np.broadcast_to(arg_fields[fn], (*outer_shape, *fshape), subok=True)
                    for fn, fshape in zip(field_names, field_shapes)
                }, shape=outer_shape).view(cls)

            broadcasted.append(arg)

        return broadcasted

    @staticmethod
    def annotations_as_metadata(cls, fields):
        """Attrs field transformer function to convert annotated fields to metadata.
        Used with Tensors subclasses for easy definition."""
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
        """Is a wrapper around attrs `@define` decorator which also subclasses the decorated class
        from Tensors.

        Using this decorator will also translate annotated fields to metadata.
        Annotated fields are expected to have a dictionary metadata with `dtype` and `shape`
        defined.

        All arguments available for `@define` are forwarded, with the exception of the following:
        - slots: forced to False due to Tensors being a subclass of np.ndarray
        - init: forced to False due to Tensors having a dedicated constructor which
            handles expected attr wrapped subclasses
        - on_setattr: makes sure setting attributes are mapped to the underlying np.ndarray fields.
            Adding more is possible as attrs supports this
        - field_transformer: default field transformer to translate annotated fields to metadata.
            Adding more is possible and works in the same manner as on_setattr
        """
        if 'slots' in kwargs or 'init' in kwargs:
            raise TypeError("Slots and init are forced to False for Tensors subclasses.")

        def subclass(cls):
            if issubclass(cls, Tensors):
                return cls
            return type(cls.__name__, (Tensors,) + cls.__bases__, dict(cls.__dict__))

        def decorate(maybe_cls, chained_field_transformer):
            new_class = define(subclass(maybe_cls), **kwargs,
                init=False, slots=False,
                on_setattr=[*kwargs.get('on_setattr', []), on_setattr],
                field_transformer=chained_field_transformer)

            new_class._dtype = new_class.__infer_dtype__()

            return new_class

        chained_field_transformer = \
            Tensors.annotations_as_metadata if "field_transformer" not in kwargs else \
            lambda cls, fields: kwargs["field_transformer"](cls,
                Tensors.annotations_as_metadata(cls, fields))

        if not maybe_cls:
            return lambda maybe_cls: decorate(maybe_cls, chained_field_transformer)

        return decorate(maybe_cls, chained_field_transformer)
