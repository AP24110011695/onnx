# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

from onnx.reference.op_run import OpRun


class Sum(OpRun):
    def _run(self, *args):
        # float16 and bfloat16 accumulate in float32: sequential float16
        # addition saturates early and ends in inf even when the true result
        # is representable (e.g. 60000 + 60000 - 60000).
        low_precision = args[0].dtype in (np.float16, ml_dtypes.bfloat16)
        acc = [a.astype(np.float32) if low_precision else a for a in args]
        return (sum(acc).astype(args[0].dtype),)
