import numpy

from chainer import cuda
from chainer import mic
from chainer import function
from chainer import utils
from chainer.utils import type_check
from time import time


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()
    _mode = libcudnn.CUDNN_ACTIVATION_RELU


class ReLU(function.Function):

    """Rectified Linear Unit."""
    # TODO(beam2d): Implement in-place version.

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward_cpu(self, x):
        start = time()
        y = utils.force_array(numpy.maximum(x[0], 0, dtype=x[0].dtype)),
        end = time() - start
        with open("/home/minhtri/workspace/numpy_test/workspace/log/log6.txt","a") as file_log:
            file_log.write("max of relu function time(forward cpu): {} \n".format(end))
        return y

    def forward_gpu(self, x):
        if (cuda.cudnn_enabled and self.use_cudnn and
                (_cudnn_version >= 3000 or x[0].dtype != numpy.float16)):
            y = cudnn.activation_forward(x[0], _mode)
            self.y = y
        else:
            y = cuda.cupy.maximum(x[0], 0)
        return y,

    def forward_mic(self, x):
        with open("/home/minhtri/workspace/numpy_test/workspace/log/log7.txt","a") as file_log: 
            file_log.write("forward relu start \n")
        start = time()
        y = mic.micpy.maximum(x[0], 0),
        end = time() - start
        with open("/home/minhtri/workspace/numpy_test/workspace/log/log6.txt","a") as file_log:
            file_log.write("max of relu function time(forward mic): {} \n".format(end))
        with open("/home/minhtri/workspace/numpy_test/workspace/log/log7.txt","a") as file_log: 
            file_log.write("forward relu end \n")
        return y

    def backward_cpu(self, x, gy):
        start = time()
        y = utils.force_array(gy[0] * (x[0] > 0)),
        end = time() - start
        with open("/home/minhtri/workspace/numpy_test/workspace/log/log6.txt","a") as file_log:
            file_log.write("* operate in y of relu function time(backward cpu): {} \n".format(end))
        return y

    def backward_gpu(self, x, gy):
        if (cuda.cudnn_enabled and self.use_cudnn and
                (_cudnn_version >= 3000 or x[0].dtype != numpy.float16)):
            gx = cudnn.activation_backward(x[0], self.y, gy[0], _mode)
        else:
            gx = cuda.elementwise(
                'T x, T gy', 'T gx',
                'gx = x > 0 ? gy : (T)0',
                'relu_bwd')(x[0], gy[0])
        return gx,

    def backward_mic(self, x, gy):
        with open("/home/minhtri/workspace/numpy_test/workspace/log/log7.txt","a") as file_log: 
            file_log.write("backward relu start \n")
        start = time()
        y = (gy[0] * (x[0] > 0)),
        end = time() - start
        with open("/home/minhtri/workspace/numpy_test/workspace/log/log6.txt","a") as file_log:
            file_log.write("* operate in y of relu function time(backward mic): {} \n".format(end))
        with open("/home/minhtri/workspace/numpy_test/workspace/log/log7.txt","a") as file_log: 
            file_log.write("backward relu end \n")
        return y


def relu(x, use_cudnn=True):
    """Rectified Linear Unit function :math:`f(x)=\\max(0, x)`.

    Args:
        x (~chainer.Variable): Input variable.
        use_cudnn (bool): If ``True`` and cuDNN is enabled, then this function
            uses cuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return ReLU(use_cudnn)(x)
