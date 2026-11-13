# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

from onnx.reference.op_run import OpRun


class CumSum(OpRun):
    def _run(self, x, axis, exclusive=None, reverse=None):
        axis = np.asarray(axis)
        if axis.ndim != 0:
            raise ValueError(f"Axis must be a rank-0 tensor, got `{axis.ndim}`.")
        # float16 and bfloat16 accumulate in float32: sequential float16
        # accumulation saturates below still-representable partial sums
        # (2048 + 1 rounds back to 2048) and drifts away from the correctly
        # rounded partial sums before it overflows.
        low_precision = x.dtype in (np.float16, ml_dtypes.bfloat16)
        xc = x.astype(np.float32) if low_precision else x
        if reverse:
            rev_indices = [slice(0, s) for s in xc.shape]
            rev_indices[axis] = slice(None, None, -1)
            xc = xc[tuple(rev_indices)]
        if exclusive:
            indices_c = [slice(0, s) for s in xc.shape]
            indices_d = [slice(0, s) for s in xc.shape]
            indices_c[axis] = slice(0, -1)
            indices_d[axis] = slice(1, xc.shape[axis])
            res = np.zeros(xc.shape, dtype=xc.dtype)
            np.cumsum(xc[tuple(indices_c)], axis=axis, out=res[tuple(indices_d)])
        else:
            res = np.cumsum(xc, axis=axis, dtype=xc.dtype)
        if reverse:
            res = res[tuple(rev_indices)]
        return (res.astype(x.dtype),)
