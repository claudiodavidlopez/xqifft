"""Array types shared by the modules of the package."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.floating[Any]]
type ComplexArray = NDArray[np.complexfloating[Any, Any]]
type IndexArray = NDArray[np.intp]
