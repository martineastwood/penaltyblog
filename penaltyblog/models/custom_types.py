from typing import Dict, Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

GoalInput = Union[Sequence[int], NDArray[np.int_], pd.Series]
TeamInput = Union[Sequence[str], NDArray[np.str_], pd.Series]
WeightInput = Union[float, Sequence[float], NDArray[np.float_], pd.Series]
ParamsOutput = Dict[str, float]
