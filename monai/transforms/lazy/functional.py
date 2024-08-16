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

from typing import Any, Sequence, Union, Tuple, Optional

import copy

import itertools as it

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_tensor import MetaTensor, get_track_meta

from monai.data.utils import to_affine_nd
from monai.transforms.lazy.resampler import resample
from monai.transforms.lazy.utils import (
    affine_from_pending,
    combine_transforms,
    is_compatible_apply_kwargs,
    kwargs_from_pending,
    MetaMatrix, Grid, Matrix, is_matrix_shaped, is_grid_shaped, AffineMatrix,
)
from monai.utils import LazyAttr

__all__ = ["apply_pending", "extents_from_shape", "shape_from_extents", "is_matrix_shaped",
           "is_grid_shaped", "MetaMatrix"]

from monai.utils import LazyAttr


def apply_pending(
        data: torch.Tensor | MetaTensor,
        pending: list | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        dtype: Any | None = np.float64,
        align_corners: bool | None = None,
        track_meta: bool | None = True
):
    """
    This method applies pending transforms to `data` tensors.

    Args:
        data: A torch Tensor or a monai MetaTensor.
        pending: pending transforms. This must be set if data is a Tensor, but is optional if data is a MetaTensor.
        mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
            Interpolation mode to calculate output values. Defaults to None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
            and the value represents the order of the spline interpolation.
            See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            When `mode` is an integer, using numpy/cupy backends, this argument accepts
            {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
            See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
        dtype: data type for resampling computation. Defaults to ``float64``.
            If ``None``, use the data type of input data`.
        align_corners: Geometrically, we consider the pixels of the input as squares rather than points, when using
            the PyTorch resampling backend. Defaults to ``None``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        track_meta: Whether the operations that are applied should be tracked (on the MetaTensor if
            the tensor is a MetaTensor)
    """
    if isinstance(data, MetaTensor) and pending is None:
        pending = data.pending_operations.copy()
        data.clear_pending_operations()
    pending = [] if pending is None else pending

    if not pending:
        if isinstance(data, MetaTensor):
            return data
        return data, []

    cumulative_xform = affine_from_pending(pending[0])
    cur_kwargs = kwargs_from_pending(pending[0].metadata)
    override_kwargs: dict[str, Any] = {}
    if mode is not None:
        override_kwargs[LazyAttr.INTERP_MODE] = mode
    if padding_mode is not None:
        override_kwargs[LazyAttr.PADDING_MODE] = padding_mode
    if align_corners is not None:
        override_kwargs[LazyAttr.ALIGN_CORNERS] = align_corners
    override_kwargs[LazyAttr.OUT_DTYPE] = data.dtype if dtype is None else dtype

    for p in pending[1:]:
        new_kwargs = kwargs_from_pending(p)
        if not is_compatible_apply_kwargs(cur_kwargs, new_kwargs):
            # carry out an intermediate resample here due to incompatibility between arguments
            _cur_kwargs = cur_kwargs.copy()
            _cur_kwargs.update(override_kwargs)
            data = resample(data, cumulative_xform, _cur_kwargs)
        next_matrix = affine_from_pending(p)
        cumulative_xform = combine_transforms(cumulative_xform, next_matrix)
        cur_kwargs.update(new_kwargs)

    # carry out any final resample that is required
    cur_kwargs.update(override_kwargs)
    data = resample(data, cumulative_xform, cur_kwargs)
    if isinstance(data, MetaTensor):
        data.clear_pending_operations()
        # TODO: at present, the resample and the modified Affine that it calls update .affine
        # on the MetaTensor. Is this the right approach or should we do it here?
        # data.affine = data.affine @ to_affine_nd(3, cumulative_xform)
        if track_meta:
            for p in pending:
                data.push_applied_operation(p)
        return data
        # return data, None

    return data, pending


def extents_from_shape(
        shape: Sequence[int],
        dtype=torch.float32
):
    """
    This method calculates a set of extents given a shape. Each extent is a point in a coordinate
    system that can be multiplied with a homogeneous matrix. As such, extents for 2D data have
    three values, and extends for 3D data have four values.

    For shapes representing 2D data, this is an array of four extents, for shape s:
     - (0, 0, 1), (0, s[1], 1), (s[0], 0, 1), (s[0], s[1], 1).

    For shapes representing 3D data, this is an array of eight extents, representing a cuboid:
     - (0, 0, 0, 1), (0, 0, s[2], 1), (0, s[1], 0, 1), (0, s[1], s[2], 1),
     - (s[0], 0, 0, 1), (s[0], 0, s[2], 1), (s[0], s[1], 0, 1), (s[0], s[1], s[2], 1)

    Args:
         shape: A shape from a numpy array or tensor
         dtype: The dtype to use for the resulting extents

    Returns:
        An array of arrays representing the shape extents
    """
    extents = [[0, shape[i]] for i in range(1, len(shape))]

    extents = it.product(*extents)
    # return [torch.as_tensor(e + (1,), dtype=dtype) for e in extents]
    return [np.asarray(e + (1,), dtype=dtype) for e in extents]


def shape_from_extents(
        src_shape: Sequence,
        extents: Union[Sequence[np.ndarray], Sequence[torch.Tensor], np.ndarray, torch.Tensor]
):
    """
    This method, given a sequence of homogeneous vertices representing the corners of a rectangle
    or cuboid, will calculate the resulting shape values from those extents.

    Args:
        src_shape: The shape into which the resulting spatial shape values will be written. Note
                   that initial shape value is appended to the spatial shape components.
        extents: The extents from which the spatial shape values should be calculated

    Returns:
        A tuple composed of the first element of `src_shape` with the spatial shape values appended
        to it.
    """
    if isinstance(extents, (list, tuple)):
        if isinstance(extents[0], np.ndarray):
            extents_ = np.asarray(extents)
        else:
            extents_ = torch.stack(extents)
            extents_ = extents_.numpy()
    else:
        if isinstance(extents, np.ndarray):
            extents_ = extents
        else:
            extents_ = extents.numpy()

    mins = extents_.min(axis=0)
    maxes = extents_.max(axis=0)
    values = np.round(maxes - mins).astype(int)[:-1].tolist()
    return (src_shape[0],) + tuple(values)


def apply_align_corners(matrix, spatial_size, op):
    """
    TODO: ensure that this functionality is correct and produces the same result as the existing ways of handling align corners
    """
    inflated_spatial_size = tuple(s + 1 for s in spatial_size)
    scale_factors = tuple(s / i for s, i in zip(spatial_size, inflated_spatial_size))
    scale_mat = op(scale_factors)
    # scale_mat = scale_mat.double()
    return matmul(scale_mat, matrix)


# this will conflict with PR Replacement Apply and Resample #5436
def matmul(
        left: Union[MetaMatrix, Grid, Matrix, NdarrayOrTensor],
        right: Union[MetaMatrix, Grid, Matrix, NdarrayOrTensor]
):
    matrix_types = (MetaMatrix, Grid, Matrix, torch.Tensor, np.ndarray)

    if not isinstance(left, matrix_types):
        raise TypeError(f"'left' must be one of {matrix_types} but is {type(left)}")
    if not isinstance(right, matrix_types):
        raise TypeError(f"'second' must be one of {matrix_types} but is {type(right)}")

    left_ = left.matrix if isinstance(left, MetaMatrix) else left
    right_ = right.matrix if isinstance(right, MetaMatrix) else right

    # TODO: it might be better to not return a metamatrix, unless we pass in the resulting
    # metadata also
    put_in_metamatrix = isinstance(left, MetaMatrix) or isinstance(right, MetaMatrix)

    put_in_grid = isinstance(left, Grid) or isinstance(right, Grid)

    put_in_matrix = isinstance(left, Matrix) or isinstance(right, Matrix)
    put_in_matrix = False if put_in_grid is True else put_in_matrix

    promote_to_tensor = not (isinstance(left_, np.ndarray) and isinstance(right_, np.ndarray))

    left_raw = left_.data if isinstance(left_, (Matrix, Grid)) else left_
    right_raw = right_.data if isinstance(right_, (Matrix, Grid)) else right_

    if promote_to_tensor:
        left_raw = torch.as_tensor(left_raw)
        right_raw = torch.as_tensor(right_raw)

    if isinstance(left_, Grid):
        if isinstance(right_, Grid):
            raise RuntimeError("Unable to matrix multiply two Grids")
        else:
            result = matmul_grid_matrix(left_raw, right_raw)
    else:
        if isinstance(right_, Grid):
            result = matmul_matrix_grid(left_raw, right_raw)
        else:
            result = matmul_matrix_matrix(left_raw, right_raw)

    if put_in_grid:
        result = Grid(result)
    elif put_in_matrix:
        result = Matrix(result)

    if put_in_metamatrix:
        result = MetaMatrix(result)

    return result


def matmul_matrix_grid(
        left: NdarrayOrTensor,
        right: NdarrayOrTensor
):
    if not is_matrix_shaped(left):
        raise ValueError(f"'left' should be a 2D or 3D homogenous matrix but has shape {left.shape}")

    if not is_grid_shaped(right):
        raise ValueError(
            "'right' should be a 3D array with shape[0] == 2 or a "
            f"4D array with shape[0] == 3 but has shape {right.shape}"
        )

    # flatten the grid to take advantage of torch batch matrix multiply
    right_flat = right.reshape(right.shape[0], -1)
    result_flat = left @ right_flat
    # restore the grid shape
    result = result_flat.reshape((-1,) + result_flat.shape[1:])
    return result


def matmul_grid_matrix(left: NdarrayOrTensor, right: NdarrayOrTensor):
    if not is_grid_shaped(left):
        raise ValueError(
            "'left' should be a 3D array with shape[0] == 2 or a "
            f"4D array with shape[0] == 3 but has shape {left.shape}"
        )

    if not is_matrix_shaped(right):
        raise ValueError(f"'right' should be a 2D or 3D homogenous matrix but has shape {right.shape}")

    try:
        inv_matrix = torch.inverse(right)
    except RuntimeError:
        # the matrix is not invertible, so we will have to perform a slow grid to matrix operation
        return matmul_grid_matrix_slow(left, right)

    # invert the matrix and swap the arguments, taking advantage of
    # matrix @ vector == vector_transposed @ matrix_inverse
    return matmul_matrix_grid(inv_matrix, left)


def matmul_grid_matrix_slow(left: NdarrayOrTensor, right: NdarrayOrTensor):
    if not is_grid_shaped(left):
        raise ValueError(
            "'left' should be a 3D array with shape[0] == 2 or a "
            f"4D array with shape[0] == 3 but has shape {left.shape}"
        )

    if not is_matrix_shaped(right):
        raise ValueError(f"'right' should be a 2D or 3D homogenous matrix but has shape {right.shape}")

    flat_left = left.reshape(left.shape[0], -1)
    result_flat = torch.zeros_like(flat_left)
    for i in range(flat_left.shape[1]):
        vector = flat_left[:, i][None, :]
        result_vector = vector @ right
        result_flat[:, i] = result_vector[0, :]

    # restore the grid shape
    result = result_flat.reshape((-1,) + result_flat.shape[1:])
    return result


def matmul_matrix_matrix(left: NdarrayOrTensor, right: NdarrayOrTensor):
    return left @ right


def lazily_apply_op(
        tensor,
        op,
        lazy_evaluation,
        track_meta=True
) -> Union[MetaTensor, Tuple[torch.Tensor, Optional[MetaMatrix]]]:
    """
    This function is intended for use only by developers of spatial functional transforms that
    can be lazily executed.

    This function will immediately apply the op to the given tensor if `lazy_evaluation` is set to
    False. Its precise behaviour depends on whether it is passed a Tensor or MetaTensor:


    If passed a Tensor, it returns a tuple of Tensor, MetaMatrix:
     - if the operation was applied, Tensor, None is returned
     - if the operation was not applied, Tensor, MetaMatrix is returned

    If passed a MetaTensor, only the tensor itself is returned

    Args:
          tensor: the tensor to have the operation lazily applied to
          op: the MetaMatrix containing the transform and metadata
          lazy_evaluation: a boolean flag indicating whether to apply the operation lazily
    """

    if isinstance(tensor, MetaTensor):
        tensor.push_pending_operation(op)
        if lazy_evaluation is False:
            response = apply_pending(tensor, track_meta=track_meta)
            result, pending = response if isinstance(response, tuple) else (response, None)
            # result, pending = apply_transforms(tensor, track_meta=track_meta)
            return result
        else:
            return tensor
    else:
        if lazy_evaluation is False:
            response = apply_pending(tensor, [op], track_meta=track_meta)
            result, pending = response if isinstance(response, tuple) else (response, None)
            # result, pending = apply_transforms(tensor, [op], track_meta=track_meta)
            return (result, op) if get_track_meta() is True else result
        else:
            return (tensor, op) if get_track_meta() is True else tensor


def invert(
        data: Union[torch.tensor, MetaTensor],
        lazy_evaluation=True
):
    metadata: MetaMatrix = data.applied_operations.pop()
    inv_metadata = copy.deepcopy(metadata)
    inv_metadata.invert()
    return lazily_apply_op(data, inv_metadata, lazy_evaluation, False)
