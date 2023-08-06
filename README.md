<p align="center">
    <picture>
        <img src="./docs/_static/logo_outlined.png" alt="attensors" />
    </picture>
</p>

Using attrs and numpy to define clear data structures containing multiple tensors.

## Overview

Attensors was born out of the need to work with multiple input data for machine learning models, while keeping things tidy and having self-documented code.

Working with mixed input data, we know the **shape** of each tensor component of a single input. This is often times found specified in intermediary preprocessing or data loading mechanisms (e.g. dataset schemas, tensor specs). All these tensors would eventually gain the same **"prefix" shape** referring global dimensions (such as a `batch_size` or a `sequence_size`).

As problems and models get more complicated, so, too, do their inputs. These collections of input tensors are almost never defined as separate entities, instead being grouped in generic data structures such as dictionaries, lists or tuples.

Through this package we propose the combined usage of `attrs` and `numpy`'s structured arrays to provide easy definition of tensor collections and intuitive means to work with them.

Furthermore, we can also make use of type hints (specifically `Annotated`) to provide the metadata describing the tensors' dtypes and shapes.

## Quick example

In general, the minimum types we need to import are the `tensors` decorator from our package and `Annotated` from typing if we plan to define fields via annotations.

```python
from attensors import tensors
from typing import Annotated
```

If we were to take for example the New York Real Estate Data as in [this](https://rosenfelder.ai/multi-input-neural-network-pytorch/) tutorial, there are multiple ways we could define our multi-tensor:

```python
# Following the tutorial, 2 tensors are needed:
# one for image data, one for tabular data
@tensors
class NYRealEstateData:
    image: Annotated[np.ndarray, {"dtype": np.float32, "shape": (3, 224, 224)}]
    tabular: Annotated[np.ndarray, {"dtype": np.float32, "shape": (5,)}]

# If we'd encode each scalar feature separately, we could define it as such
@tensors
class NYRealEstateData:
    image: Annotated[np.ndarray, {"dtype": np.float32, "shape": (3, 224, 224)}]
    latitude: float
    longitude: float
    zpid: int
    beds: int
    baths: int

# We could also group latitude and longitude separately and nest types
@tensors
class Coordinates:
    latitude: float
    longitude: float

@tensors
class NYRealEstateData:
    image: Annotated[np.ndarray, {"dtype": np.float32, "shape": (3, 224, 224)}]
    location: Coordinates
    zpid: int
    beds: int
    baths: int
```

Instantiating can be done by simply providing the data, or via numpy styled routines. In the following contexts we consider the first definition provided above and 2 samples given with their respective tensors `i1`, `t1` and  `i2`, `t2`.

```python
>>> sample = NYRealEstateData(image=[i1, i2], tabular=[t1, t2])
```

Since `NYRealEstateData` is actually a `numpy.ndarray` subclass, you could also define using numpy

```python
>>> sample = np.array([(i1, t1), (i2, t2)], dtype=NYRealEstateData._dtype).view(NYRealEstateData)
```

Generating dummy data can be accomplished via shortcutted numpy routines.

```python
>>> sample = NYRealEstateData.empty((2,))
```

All above examples will result in a sample with a prefix shape of (2,).

```
>>> sample.shape
(2,)
>>> sample.image.shape
(2,3,224,224)
>>> sample.tabular.shape
(2,5)
```

Of course, indexing, [universal functions](#universal-functions) and classic array manipulation routines such as `reshape`, `stack`, `concatenate`  are also supported.
You can read more about this in the [documentation](#documentation).

## Documentation

### Type Definition

`Tensors` is the baseclass used for defining tensor collection types. It is created as a subclass of numpy's `ndarray`, using the functionality of structured arrays. On top of this, some specific rules are implemented to handle broadcasting, universal functions and shortcutting some numpy routines given the underlying dtype definition given by provided attributes.

Defining `Tensors` types is done via the `@tensors` decorator which is a wrapper over attrs' `@define`.
The decorator provides the same functionality, however it also subclasses the decorated class from `Tensors`.

All arguments available for `@define` are forwarded, with the exception of the following:
- `slots`: forced to False due to Tensors being a subclass of np.ndarray
- `init`: forced to False due to Tensors having a dedicated constructor which
    handles expected attr wrapped subclasses
- `on_setattr`: makes sure setting attributes are mapped to the underlying np.ndarray fields.
    Adding more is possible as attrs supports this
- `field_transformer`: default field transformer to translate annotated fields to metadata.
    Adding more is possible and works in the same manner as on_setattr

Using this decorator will also translate annotated fields to metadata.
Annotated fields are expected to have a dictionary metadata with `dtype` and `shape`
defined.

If fields are defined through type hints, but not using `Annotated` the type will be considered as the dtype of the underlying tensor and it's shape will be ().

Fields can also be defined using attrs' `field` function.

### Instantiation

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

### Indexing and attributes

Classes defined with `@tensors`, in the style of attrs, will provide attributes corresponding to the underlying ndarray structured array fields. Besides this, classic ndarray attributes such as `shape` and `dtype` will also be available. Types defined with the decorator will also contain a `_dtype` attribute corresponding to the inferred numpy dtype, which is used upon instantiation.

Indexing works in the same manner as it does with structured numpy arrays, however some type casting may occur, in the following manner:
- Any indexing that would result in an unstructured array will be cast to ndarray
- Any indexing that would result in a structured array which has a different dtype than the class, will be cast to Tensors
- Any indexing that would extract a single element (type numpy.void) will be cast to the source class.

### Broadcast rules

Additional broadcasting rules have been implemented through the class method `broadcast_tensors`.

`Tensors` types cannot be broadcast together unless they have the same field identifiers.

Scalars and non-structured ndarrays broadcast with `Tensors` types will be brought to that type's structure by broadcasting independently with each tensor component.

`Tensors` with the same fields will broadcast the base shape of each coresponding component and then also broadcast the prefix shape as well.

### Universal functions

Numpy provides an API for the multitude of mathematical operations via [universal functions](https://numpy.org/doc/stable/user/basics.ufuncs.html#ufuncs-basics). However, these do not work for structured arrays, perahps due to the impossibility to define a universal standard approach for these.

In this package, universal functions are implemented for given types that can be broadcast following the above rules and will work as such:

Any universal function called on `Tensors` types will essentially be called on each underlying tensor component.

The result(s) will be one or more `Tensors` types, having the same field identifiers, resulting dtypes based on the operation and broadcast shapes.