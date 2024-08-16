# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms import ResizeWithPadOrCropd
import monai.transforms.croppad.dictionary as lcd
import monai.transforms.croppad.old_dictionary as tcd
from tests.utils import TEST_NDARRAYS_ALL, pytorch_after

TEST_CASES = [
    [{"keys": "img", "spatial_size": [15, 8, 8], "mode": "constant"}, {"img": np.zeros((3, 8, 8, 4))}, (3, 15, 8, 8)],
    [
        {"keys": "img", "spatial_size": [15, 4, 8], "mode": "constant", "method": "end", "constant_values": 1},
        {"img": np.zeros((3, 8, 8, 4))},
        (3, 15, 4, 8),
    ],
    [{"keys": "img", "spatial_size": [15, 4, -1], "mode": "constant"}, {"img": np.zeros((3, 8, 8, 4))}, (3, 15, 4, 4)],
    [
        {"keys": "img", "spatial_size": [15, 4, -1], "mode": "reflect" if pytorch_after(1, 11) else "constant"},
        {"img": np.zeros((3, 8, 8, 4))},
        (3, 15, 4, 4),
    ],
    [
        {"keys": "img", "spatial_size": [-1, -1, -1], "mode": "reflect" if pytorch_after(1, 11) else "constant"},
        {"img": np.zeros((3, 8, 8, 4))},
        (3, 8, 8, 4),
    ],
]


class TestResizeWithPadOrCropd(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_pad_shape(self, input_param, input_data, expected_val):
        for p in TEST_NDARRAYS_ALL:
            if isinstance(p(0), torch.Tensor) and (
                "constant_values" in input_param or input_param["mode"] == "reflect"
            ):
                continue
            padcropper = ResizeWithPadOrCropd(**input_param)
            data = input_data.copy()
            data["img"] = p(input_data["img"])
            result = padcropper(data)
            np.testing.assert_allclose(result["img"].shape, expected_val)
            inv = padcropper.inverse(result)
            for k in data:
                self.assertTupleEqual(inv[k].shape, data[k].shape)

            l_input_param = input_param.copy()
            l_input_param["padding_mode"] = input_param["mode"]
            del l_input_param["mode"]
            l_padcropper = lcd.ResizeWithPadOrCropd(lazy=False, **l_input_param)
            l_data = input_data.copy()
            l_data["img"] = p(input_data["img"])
            l_result = l_padcropper(l_data)
            self.assertSequenceEqual(l_result["img"].shape, expected_val)
            self.assertTrue(torch.allclose(l_result["img"], result["img"]))

    def test_compare_trad_and_lazy(self):
        # print(len(entries))
        spatial_size = (40, 56, 40)
        tcrop = tcd.ResizeWithPadOrCropd(keys=('image',), spatial_size=spatial_size)
        lcrop = lcd.ResizeWithPadOrCropd(keys=('image',), spatial_size=spatial_size, lazy=False)
        rows = 40

        s_img = np.arange(0, 48 * 48 * 48, dtype=np.int32).reshape((1, 48, 48, 48))
        data = {'image': s_img}
        image = data['image']
        t_image = tcrop(data)['image']
        l_image = lcrop(data)['image']
        print(image.dtype, t_image.dtype, l_image.dtype)


if __name__ == "__main__":
    unittest.main()
