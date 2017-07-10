import numpy
import micpy

from chainer import cuda

__device = None
ndarray = micpy.ndarray
available = micpy.device.check_available()


def get_device(*args):
    """Gets the device from a device object, an ID integer or an array object.

    This is a convenient utility to select a correct device if the type of
    ``arg`` is unknown (i.e., one can use this function on arrays that may be
    on CPU or MIC). The returned device object supports the context management
    protocol of Python for the *with* statement.

    Args:
        args: Values to specify a MIC device. The first device object, integer
            or :class:`cupy.ndarray` object is used to select a device.
            If it is a device object, it is returned. If it is an integer,
            the corresponding device is returned. If it is a MICPy array,
            the device on which this array reside is returned. If any
            arguments are neither integers nor MICPy arrays, a dummy device
            object representing CPU is returned.

    Returns:
        Device object specified by given ``args``.

    .. seealso::
       See :class:`cupy.cuda.Device` for the device selection not by arrays.

    """
    for arg in args:
        if arg is None:
            device_id = micpy.device.get_device_id()
            return micpy.device.Device(device_id)
        if isinstance(arg, (int, long)):
            return micpy.device.Device(arg)
        if isinstance(arg, micpy.ndarray):
            device_id = arg.device.device_id
            return micpy.device.Device(device_id)
        if isinstance(arg, micpy.device.Device):
            return arg
    return cuda.DummyDevice


def check_mic_available():
    if not available:
        raise RuntimeError("No MIC device is availabe right now")


def to_mic(array, device=None):
    """Copies the given CPU array to specified device.

    Args:
        array: Array to be sent to Xeon Phi device
        device (int): Device specifier

    Returns:
        micpy.ndarray: Array on Xeon Phi
        If ``array`` is already on Xeon Phi then this funtion just returns
        ``array`` without performing any copy
    """
    #TODO(superbo): check logic
    check_mic_available()
    if isinstance(array, numpy.ndarray):
        with get_device(device):
            a = micpy.asarray(array, update_device=True)
            micpy.sync()
            a.array = None
            return a
    if isinstance(array, micpy.ndarray):
        return array


def to_cpu(array, stream=None):
    """Copies the given MIC array to host CPU.

    Args:
        array: Array to be sent to CPU.
        stream (micpy.stream): MIC stream.

    Returns:
        numpy.ndarray: Array on CPU.

        If given ``array`` is already on CPU, then this function just returns
        ``array`` without performing any copy.

    """
    if isinstance(array, numpy.ndarray):
        return array
    if isinstance(array, micpy.ndarray):
        check_mic_available()
        if stream is None:
            stream = get_device(array).stream

        if array.array is None:
            cpu_arr = numpy.empty(array.shape, array.dtype)
        else:
            cpu_arr = array.array

        host_ptr = cpu_arr.ctypes.get_data()
        stream.transfer_device2host(array._device_ptr, host_ptr, array._nbytes)
        stream.sync()

        return cpu_arr
    raise TypeError(
        'The array sent to cpu must be micpy.ndarray or numpy.ndarray.'
        '\nActual type: {0}.'.format(type(array)))


def get_array_module(*args):
    for arg in args:
        if isinstance(arg, micpy.ndarray):
            return micpy

    return numpy
