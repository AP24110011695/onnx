# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Sum(Base):
    @staticmethod
    def export() -> None:
        data_0 = np.array([3, 0, 2]).astype(np.float32)
        data_1 = np.array([1, 3, 4]).astype(np.float32)
        data_2 = np.array([2, 6, 6]).astype(np.float32)
        result = np.array([6, 9, 12]).astype(np.float32)
        node = onnx.helper.make_node(
            "Sum",
            inputs=["data_0", "data_1", "data_2"],
            outputs=["result"],
        )
        expect(
            node,
            inputs=[data_0, data_1, data_2],
            outputs=[result],
            name="test_sum_example",
        )

        node = onnx.helper.make_node(
            "Sum",
            inputs=["data_0"],
            outputs=["result"],
        )
        expect(node, inputs=[data_0], outputs=[data_0], name="test_sum_one_input")

        result = np.add(data_0, data_1)
        node = onnx.helper.make_node(
            "Sum",
            inputs=["data_0", "data_1"],
            outputs=["result"],
        )
        expect(
            node, inputs=[data_0, data_1], outputs=[result], name="test_sum_two_inputs"
        )

    @staticmethod
    def export_float16_large_values() -> None:
        # Sequential float16 addition saturates early and ends in inf even
        # when the true result is representable: 60000 + 60000 overflows and
        # inf - 60000 stays inf. Expected outputs are computed in float32.
        data_0 = np.full((2, 2), 60000).astype(np.float16)
        data_1 = np.full((2, 2), 60000).astype(np.float16)
        data_2 = np.full((2, 2), -60000).astype(np.float16)
        result = (
            data_0.astype(np.float32)
            + data_1.astype(np.float32)
            + data_2.astype(np.float32)
        ).astype(np.float16)
        node = onnx.helper.make_node(
            "Sum",
            inputs=["data_0", "data_1", "data_2"],
            outputs=["result"],
        )
        expect(
            node,
            inputs=[data_0, data_1, data_2],
            outputs=[result],
            name="test_sum_float16_large_values",
        )
