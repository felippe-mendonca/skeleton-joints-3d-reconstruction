import numpy as np
from is_msgs.camera_pb2 import CameraCalibration
from is_msgs.common_pb2 import Tensor, DataType


def to_tensor(array):
    tensor = Tensor()
    if len(array.shape) == 0:
        return tensor
    if len(array.shape) > 2:
        raise Exception('Implemented only for one or two dimensional np.ndarray')

    dims_name = ['rows', 'cols']
    for size, name in zip(array.shape, dims_name):
        dim = tensor.shape.dims.add()
        dim.size = size
        dim.name = name

    dtype = array.dtype
    if dtype in ['int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32']:
        tensor.type = DataType.Value('INT32_TYPE')
        tensor.ints32.extend(array.ravel().tolist())
    elif dtype in ['int64', 'uint64']:
        tensor.type = DataType.Value('INT64_TYPE')
        tensor.ints64.extend(array.ravel().tolist())
    elif dtype in ['float16', 'float32']:
        tensor.type = DataType.Value('FLOAT_TYPE')
        tensor.floats.extend(array.ravel().tolist())
    elif dtype in ['float64']:
        tensor.type = DataType.Value('DOUBLE_TYPE')
        tensor.doubles.extend(array.ravel().tolist())
    else:
        pass
    
    return tensor


def to_np(tensor):
    if len(tensor.shape.dims) != 2 or tensor.shape.dims[0].name != 'rows':
        return np.array([])

    shape = (tensor.shape.dims[0].size, tensor.shape.dims[1].size)
    if tensor.type == DataType.Value('INT32_TYPE'):
        return np.array(tensor.ints32, dtype=np.int32, copy=False).reshape(shape)
    if tensor.type == DataType.Value('INT64_TYPE'):
        return np.array(tensor.ints64, dtype=np.int64, copy=False).reshape(shape)
    if tensor.type == DataType.Value('FLOAT_TYPE'):
        return np.array(tensor.floats, dtype=np.float32, copy=False).reshape(shape)
    if tensor.type == DataType.Value('DOUBLE_TYPE'):
        return np.array(tensor.doubles, dtype=np.float64, copy=False).reshape(shape)
    return np.array([])