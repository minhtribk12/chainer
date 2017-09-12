import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check
from chainer import device as devutil
from time import time

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()
    _algorithm = libcudnn.CUDNN_SOFTMAX_LOG
    _mode = libcudnn.CUDNN_SOFTMAX_MODE_CHANNEL


def logsumexp(x):
    xp = devutil.get_array_module(x)
    with open("/home/minhtri/workspace/numpy_test/workspace/log/log7.txt","a") as file_log: 
        file_log.write("max start \n")
    start = time()
    m = x.max(axis=1, keepdims=True)
    end = time() - start
    with open("/home/minhtri/workspace/numpy_test/workspace/log/log6.txt","a") as file_log:
        file_log.write("max operate on x of logsumexp function time(forward): {} \n".format(end))
    with open("/home/minhtri/workspace/numpy_test/workspace/log/log7.txt","a") as file_log: 
        file_log.write("max end \n")
    with open("/home/minhtri/workspace/numpy_test/workspace/log/log7.txt","a") as file_log: 
        file_log.write("- start \n")
    start = time()
    y = x - m
    end = time() - start
    with open("/home/minhtri/workspace/numpy_test/workspace/log/log6.txt","a") as file_log:
        file_log.write("- operate on y of logsumexp function time(forward): {} \n".format(end))
    with open("/home/minhtri/workspace/numpy_test/workspace/log/log7.txt","a") as file_log: 
        file_log.write("- end \n")
    start = time()
    xp.exp(y, out=y)
    end = time() - start
    with open("/home/minhtri/workspace/numpy_test/workspace/log/log6.txt","a") as file_log:
        file_log.write("exp operate on y of logsumexp function time(forward): {} \n".format(end))
    with open("/home/minhtri/workspace/numpy_test/workspace/log/log7.txt","a") as file_log: 
        file_log.write("sum start \n")
    start = time()
    s = y.sum(axis=1, keepdims=True)
    end = time() - start
    with open("/home/minhtri/workspace/numpy_test/workspace/log/log6.txt","a") as file_log:
        file_log.write("sum operate on y of logsumexp function time(forward): {} \n".format(end))
    with open("/home/minhtri/workspace/numpy_test/workspace/log/log7.txt","a") as file_log: 
        file_log.write("sum end \n")
    start = time()
    xp.log(s, out=s)
    end = time() - start
    with open("/home/minhtri/workspace/numpy_test/workspace/log/log6.txt","a") as file_log:
        file_log.write("log operate on s of logsumexp function time(forward): {} \n".format(end))
    with open("/home/minhtri/workspace/numpy_test/workspace/log/log7.txt","a") as file_log: 
        file_log.write("+= start \n")
    start = time()
    m += s
    end = time() - start
    with open("/home/minhtri/workspace/numpy_test/workspace/log/log6.txt","a") as file_log:
        file_log.write("+= operate on m of logsumexp function time(forward): {} \n".format(end))
    with open("/home/minhtri/workspace/numpy_test/workspace/log/log7.txt","a") as file_log: 
        file_log.write("+= end \n")
    return m


def _log_softmax(x, use_cudnn):
    if cuda.cudnn_enabled and use_cudnn and _cudnn_version >= 3000:
        xp = cuda.get_array_module(x)
        if xp != numpy:
            oz_dtype = 'd' if x.dtype == 'd' else 'f'
            one = numpy.array(1, dtype=oz_dtype).ctypes
            zero = numpy.array(0, dtype=oz_dtype).ctypes
            handle = cudnn.get_handle()
            x_cube = x.reshape(x.shape[:2] + (-1, 1))
            desc = cudnn.create_tensor_descriptor(x_cube)
            y = xp.empty_like(x)
            libcudnn.softmaxForward(
                handle, _algorithm, _mode, one.data, desc.value,
                x_cube.data.ptr, zero.data, desc.value,
                y.data.ptr)
            return y
    with open("/home/minhtri/workspace/numpy_test/workspace/log/log7.txt","a") as file_log: 
        file_log.write("logsumexp start \n")
    log_z = logsumexp(x)
    with open("/home/minhtri/workspace/numpy_test/workspace/log/log7.txt","a") as file_log: 
        file_log.write("logsumexp end \n")
    with open("/home/minhtri/workspace/numpy_test/workspace/log/log7.txt","a") as file_log: 
        file_log.write("- start \n")
    start = time()
    y = x - log_z
    end = time() - start
    with open("/home/minhtri/workspace/numpy_test/workspace/log/log6.txt","a") as file_log:
        file_log.write("- operate on y of _log_softmax function time(forward): {} \n".format(end))
    with open("/home/minhtri/workspace/numpy_test/workspace/log/log7.txt","a") as file_log: 
        file_log.write("- end \n")
    return y


class LogSoftmax(function.Function):

    """Log-softmax activation function."""

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn
        self.y = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim > 1,
        )

    def forward(self, xs):
        self.y = _log_softmax(xs[0], self.use_cudnn)
        return self.y,

    def backward(self, x, gy):
        start = time()
        xp = devutil.get_array_module(*x)
        if (xp != numpy and cuda.cudnn_enabled and self.use_cudnn and
                _cudnn_version >= 3000):
            oz_dtype = 'd' if x[0].dtype == 'd' else 'f'
            one = numpy.array(1, dtype=oz_dtype).ctypes
            zero = numpy.array(0, dtype=oz_dtype).ctypes
            handle = cudnn.get_handle()
            gx = xp.empty_like(x[0])
            gx_cube = gx.reshape(gx.shape[:2] + (-1, 1))
            desc = cudnn.create_tensor_descriptor(gx_cube)
            libcudnn.softmaxBackward(
                handle, _algorithm, _mode, one.data, desc.value,
                self.y.data.ptr, desc.value, gy[0].data.ptr, zero.data,
                desc.value, gx.data.ptr)
        else:
            gx = gy[0] - xp.exp(self.y) * gy[0].sum(axis=1, keepdims=True)
        end = time() - start
        with open("/home/minhtri/workspace/numpy_test/workspace/log/log6.txt","a") as file_log:
            file_log.write("backward of LogSoftmax time(mic): {} \n".format(end))
        return gx,


def log_softmax(x, use_cudnn=True):
    """Channelwise log-softmax function.

    This function computes its logarithm of softmax along the second axis. Let
    :math:`i = (i_1, i_2, \\dots, i_d)^{\\top}` be the d dimensional index
    array and :math:`x = f(i)` be the corresponding d dimensional input array.
    For each index :math:`i` of the input array :math:`f(i)`, it computes the
    logarithm of the probability :math:`\log p(x)` defined as

    .. math::
        p(i) = {\\exp(f(i)) \\over \\sum_{i'_2} \\exp(f(i'))},

    where :math:`i' = (i_1, i'_2, \\dots, i_d)`.

    .. math::
        p(x) = {\\exp(f(x)) \\over \\sum_{x'} \\exp(f(x'))}.

    This method is theoretically equivalent to ``log(softmax(x))`` but is more
    stable.

    .. note::
        ``log(softmax(x))`` may cause underflow when ``x`` is too small,
        because ``softmax(x)`` may returns ``0``.
        ``log_softmax`` method is more stable.

    Args:
        x (~chainer.Variable): Input variable.
        use_cudnn (bool): If ``True``, cuDNN is enabled and cuDNN ver. 3 or
            later is used, then this function uses cuDNN as the core
            implementation.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :func:`~chainer.functions.softmax`

    """
    return LogSoftmax(use_cudnn)(x)
