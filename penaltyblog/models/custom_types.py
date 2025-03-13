from typing import Dict, Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

GoalInput = Union[Sequence[int], NDArray[np.int64], pd.Series]
TeamInput = Union[Sequence[str], NDArray[np.str_], pd.Series]
WeightInput = Union[float, Sequence[float], NDArray[np.float64], pd.Series, None]
ParamsOutput = Dict[str, float]
