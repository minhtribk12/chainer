from chainer import cuda
from chainer import mic
import numpy


#Device code
GPU = cuda.available
MIC = mic.available

DummyDevice = cuda.DummyDevice


if cuda.available:
    to_device = cuda.to_gpu
elif mic.available:
    to_device = mic.to_mic


# ------------------------------------------------------------------------------
# Global states
# ------------------------------------------------------------------------------
def get_device(*args):
    """Gets the device from a device object, an ID integer or an array object.

    This is a convenient utility to select a correct device if the type of
    ``arg`` is unknown (i.e., one can use this function on arrays that may be
    on CPU or GPU). The returned device object supports the context management
    protocol of Python for the *with* statement.

    Args:
        args: Values to specify a GPU device. The first device object, integer
            or :class:`cupy.ndarray` object is used to select a device.
            If it is a device object, it is returned. If it is an integer,
            the corresponding device is returned. If it is a CuPy array,
            the device on which this array reside is returned. If any
            arguments are n_integereither integers nor CuPy arrays, a dummy device
            object representing CPU is returned.

    Returns:
        Device object specified by given ``args``.

    .. seealso::
       See :class:`cupy.cuda.Device` for the device selection not by arrays.

    """
    if cuda.available:
        return cuda.get_device(*args)
    if mic.available:
        return mic.get_device(*args)
    return DummyDevice


# ------------------------------------------------------------------------------
# ndarray allocation and copy
# ------------------------------------------------------------------------------

def to_gpu(array, device=None, stream=None):
    """Copies the given CPU array to specified GPU device.

    Args:
        array: Array to be sent to GPU.
        device: Device specifier.
        stream (cupy.cuda.Stream): CUDA stream. If not ``None``, the copy runs
            asynchronously.

    Returns:
        cupy.ndarray: Array on GPU.

        If ``array`` is already on GPU, then this function just returns
        ``array`` without performing any copy. Note that this function does not
        copy :class:`cupy.ndarray` into specified device.

    """
    if isinstance(array, (numpy.ndarray, cuda.cupy.ndarray)):
        return cuda.to_gpu(array)
    if isinstance(array, mic.micpy.ndarray):
        raise TypeError("Transfering from MIC array to GPU array is not supported")


def to_cpu(array, stream=None):
    """Copies the given GPU array to host CPU.

    Args:
        array: Array to be sent to CPU.
        stream (cupy.cuda.Stream): CUDA stream.

    Returns:
        numpy.ndarray: Array on CPU.

        If given ``array`` is already on CPU, then this function just returns
        ``array`` without performing any copy.

    """
    if isinstance(array, cuda.ndarray):
        return cuda.to_cpu(array, stream)
    if isinstance(array, mic.ndarray):
        return mic.to_mic(array, stream)
    if isinstance(array, numpy.ndarray):
        return array
    else:
        raise TypeError(
            'The array sent to cpu must be numpy.ndarray, cupy.ndarray or micpy.ndarray.'
            '\nActual type: {0}.'.format(type(array)))


def to_mic(array, stream=None):
    """Copies the given CPU array to specified MIC device.

    Args:
        array: Array to be sent to MIC.
        device: Device specifier.
        stream (micpy.stream): MIC stream. If not ``None``, the copy runs
            asynchronously.

    Returns:
        micpy.ndarray: Array on MIC.

        If ``array`` is already on MIC, then this function just returns
        ``array`` without performing any copy. Note that this function does not
        copy :class:`micpy.ndarray` into specified device.

    """
    if isinstance(array, mic.ndarray) or isinstance(array, numpy.ndarray):
        return mic.to_mic(array)
    if isinstance(cuda.ndarray):
        raise TypeError("Transfering GPU array to MIC array is not supported")


# ------------------------------------------------------------------------------
# numpy/cupy compatible coding
# ------------------------------------------------------------------------------
def get_array_module(*args):
    """Gets an appropriate one from :mod:`numpy` or :mod:`cupy`.

    This is almost equivalent to :func:`cupy.get_array_module`. The only
    difference is that this function can be used even if CUDA is not available.

    Args:
        args: Values to determine whether NumPy or CuPy should be used.

    Returns:
        module: :mod:`cupy` or :mod:`numpy` is returned based on the types of
        the arguments.

    """
    if cuda.available:
        return cuda.get_array_module(*args)
    elif mic.available:
        return mic.get_array_module(*args)
    else:
        return numpy
