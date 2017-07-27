import numpy

from chainer import cuda

__device = None

try:
    import micpy
    ndarray = micpy.ndarray
    import micpy.random
    available = True
except ImportError:
    available = False


class Device:
    """Object that represents a MIC device.

    This class provides some basic manipulations on MIC devices.

    It supports the context protocol. For example, the following code is an
    example of temporarily switching the current device::

       with Device(0):
           do_something_on_device_0()

    After the *with* statement gets done, the current device is reset to the
    original one.

    Args:
        device (int or cupy.cuda.Device): Index of the device to manipulate. Be
            careful that the device ID (a.k.a. GPU ID) is zero origin. If it is
            a Device object, then its ID is used. The current device is
            selected by default.

    Attributes:
        id (int): ID of this device.

    """

    def __init__(self, device=None):
        if device is None:
            self.id = micpy.device()
        else:
            self.id = int(device)

        self._device_stack = []

    def __int__(self):
        return self.id

    def __enter__(self):
        id = micpy.device()
        self._device_stack.append(id)
        if self.id != id:
            self.use()
        return self

    def __exit__(self, *args):
        micpy.set_device(self._device_stack.pop())

    def __repr__(self):
        return '<MIC Device %d>' % self.id

    def use(self):
        """Makes this device current.

        If you want to switch a device temporarily, use the *with* statement.

        """
        micpy.set_device(self.id)

    def synchronize(self):
        """Synchronizes the current thread to the device."""
        #TODO: enable in future
        pass

    def __richcmp__(self, other, op):
        if op == 0:
            return self.id < other.id
        if op == 1:
            return self.id <= other.id
        if op == 2:
            return self.id == other.id
        if op == 3:
            return self.id != other.id
        if op == 4:
            return self.id > other.id
        if op == 5:
            return self.id >= other.id
        return NotImplemented


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
            return Device()
        if isinstance(arg, (int, long)):
            return Device(arg)
        if isinstance(arg, micpy.ndarray):
            return Device(arg.device)
        if isinstance(arg, Device):
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
    devid = get_device(device).id
    if isinstance(array, numpy.ndarray) or array.device != devid:
        return micpy.to_mic(array, devid)
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
        return array.to_cpu()
    raise TypeError(
        'The array sent to cpu must be micpy.ndarray or numpy.ndarray.'
        '\nActual type: {0}.'.format(type(array)))


def get_array_module(*args):
    for arg in args:
        if isinstance(arg, micpy.ndarray):
            return micpy

    return numpy
