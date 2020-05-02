from json import JSONEncoder
from pathlib import PosixPath

import numpy as np


class CustomJsonEncoder(JSONEncoder):
    """Json encoder class, which handles not json serializable objects.

    Examples:
        This class can be used as a json serializer in dump json method.

        >>> import json
        >>> import numpy as np
        >>> data = {'a': np.array([1, 2, 3]), 'b': 123}
        >>> json_str = json.dumps(data, cls=CustomJsonEncoder)
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, PosixPath):
            return str(obj)

        return JSONEncoder.default(self, obj)
