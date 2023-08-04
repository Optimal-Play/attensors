<p align="center">
    <picture>
        <img src="./docs/_static/logo_outlined.png" alt="attensors" />
    </picture>
</p>

[WIP] Using attrs and numpy to define clear data structures containing multiple tensors.

## Overview

Attensors was born out of the need to work with mixed input data for machine learning models, while keeping things tidy and have code self-documented.

Working with mixed input data, we know the **shape** of each tensor component of a single input. This is often times found specified in intermediary preprocessing or data loading mechanisms (e.g. dataset schemas, tensor specs). Of course, all these tensors would have the same **"prefix" shape** referring global dimensions (such as `batch_size` or `sequence_size`).

As problems and models get more complicated, so, too, do their inputs. These collections of input tensors are almost never defined as separate entities, instead being grouped in generic data structures such as dictionaries, lists or tuples.

Through this package we propose the combined usage of `attrs` with `numpy`'s structured arrays to provide easy definition of tensor collections and intuitive means to work with them.

Furthermore, we can make use of type hints and `Annotated` to provide the metadata describing the tensors' type and shape. However, these can also be inferred if not provided.

## Example usage

In general, the minimum types we need to import are the `tensors` decorator from our package and `Annotated` from typing if we plan to define fields via annotation.

```python
from attensors import tensors
from typing import Annotated
```

If we were to take for example the New York Real Estate Data as in [this](https://rosenfelder.ai/multi-input-neural-network-pytorch/) tutorial, there are multiple ways we could define our multi-tensor:

```python
# Following the tutorial, 2 tensors are needed: one for image data, one for tabular data
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

Instantiating can be done by simply providing the data, or via numpy styled routines. In the following context we consider 2 samples given with their respective tensors `img_1`, `tab_1` and  `img_2`, `tab_2`

```python
>>> sample = NYRealEstateData(image=[img_1, img_2], tabular=[tab_1, tab_2])
```

Since `NYRealEstateData` is actually a `numpy.ndarray` subclass, you could also define using numpy

```python
>>> sample = np.array([(img_1, tab_1), (img_2, tab_2)], dtype=NYRealEstateData._dtype).view(NYRealEstateData)
```

Generating dummy data can be easily done via shortcutted numpy routines

```python
>>> sample = NYRealEstateData.empty((2,))
```

Of course, [universal functions](#universal-functions) and classic array manipulation routines such as `reshape`, `stack`, `concatenate`  are also supported.
You can read more about this in the [documentation](#documentation).

## Documentation

### Broadcast rules

TODO

### Universal functions

TODO