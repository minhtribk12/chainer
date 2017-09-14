from chainer import function
from chainer.utils import type_check
import numpy as np
import pymic as mic


def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


class LinearFunction(function.Function):

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim >= 2,
            w_type.ndim == 2,
            type_check.prod(x_type.shape[1:]) == w_type.shape[1],
        )
        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )
    #minhtribk12 - modified 13/09/17
    #Start
    def iadd_mic(self, operand1, operand2):
        # load the library with the kernel function (on the target)
        device_mic = mic.devices[0]
        library_mic = device_mic.load_library("libdgemm.so")

        # use the default stream
        stream_mic = device_mic.get_default_stream()

        # sizes of the matrices
        m = operand1.shape[0]
        n = operand1.shape[1]

        a_ = np.tile(operand2,(m,1))

        # construct some matrices
        operand1.reshape((m,n))
        a_.reshape((m,n))
        b_ = np.diag(np.ones(n)).reshape((n,n))
        
        alpha_mic = 1.0
        beta_mic = 1.0
        
        offl_a = stream_mic.bind(a_)
        offl_b = stream_mic.bind(b_)
        offl_c = stream_mic.bind(operand1)

        stream_mic.invoke(library_mic.dgemm_kernel, offl_a, offl_b, offl_c, m, n, n, alpha_mic, beta_mic)
        
        stream_mic.sync()
        
        offl_c.update_host()
        output_mic = offl_c.array
        return output_mic
    def dot_mic(self, operand1, operand2):
        a = operand1.copy()
        b = operand2.copy()
        alpha = 1.0
        beta = 0.0
        with open("./log/log7.txt","a") as file_log: 
            file_log.write("point 1 \n")
        m = a.shape[0]
        n = b.shape[1]
        k = a.shape[1]
        c = np.zeros((m,n))
        with open("./log/log7.txt","a") as file_log: 
            file_log.write("point 2 \n")
        # load the library with the kernel function (on the target)
        with open("./log/log7.txt","a") as file_log: 
            file_log.write("point 3 \n")
        device = mic.devices[0]
        # use the default stream
        stream = device.get_default_stream()        
        with open("./log/log7.txt","a") as file_log: 
            file_log.write("point 4 \n")
        library = device.load_library("libbenchmark_kernels.so")
        with open("./log/log7.txt","a") as file_log: 
            file_log.write("point 5 \n")
        # associate host arrays with device arrats
        offl_a = stream.bind(a)
        offl_b = stream.bind(b)
        offl_c = stream.bind(c)
        stream.sync()
        with open("./log/log7.txt","a") as file_log: 
            file_log.write("point 6 \n")
        stream.invoke(library.dgemm_kernel,
              offl_a, offl_b, offl_c,
              m, n, k, alpha, beta)
        with open("./log/log7.txt","a") as file_log: 
            file_log.write("point 7 \n")
        stream.sync()
        with open("./log/log7.txt","a") as file_log: 
            file_log.write("point 8 \n")
        offl_c.update_host()
        with open("./log/log7.txt","a") as file_log: 
            file_log.write("point 9 \n")
        stream.sync()
        #stream.deallocate_device_memory(offl_a._device_ptr)
        #stream.deallocate_device_memory(offl_b._device_ptr)
        #stream.deallocate_device_memory(offl_c._device_ptr)
        #stream.sync()
        #del stream
        output_mic = c.copy()
        return output_mic
    #End
    def forward(self, inputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        #y = x.dot(W.T)
        #u = self.dot_mic(x, (W.T))
        #y = u.astype(x.dtype, copy=False)
        with open("./log/log7.txt","a") as file_log: 
            file_log.write("dot start \n")
        y = self.dot_mic(x,(W.T)).astype(x.dtype, copy=False)
        with open("./log/log7.txt","a") as file_log: 
            file_log.write("dot stop \n")
        if len(inputs) == 3:
            b = inputs[2]
            y += b
            #u = self.iadd_mic(y,b)
        return y,

    def backward(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        gy = grad_outputs[0]

        gx = gy.dot(W).astype(x.dtype, copy=False).reshape(inputs[0].shape)
        gW = gy.T.dot(x).astype(W.dtype, copy=False)
        if len(inputs) == 3:
            gb = gy.sum(0)
            return gx, gW, gb
        else:
            return gx, gW


def linear(x, W, b=None):
    """Linear function, or affine transformation.

    It accepts two or three arguments: an input minibatch ``x``, a weight
    matrix ``W``, and optionally a bias vector ``b``. It computes
    :math:`Y = xW^\\top + b`.

    Args:
        x (~chainer.Variable): Input variable. Its first dimension is assumed
            to be the *minibatch dimension*. The other dimensions are treated
            as concatenated one dimension whose size must be ``N``.
        W (~chainer.Variable): Weight variable of shape ``(M, N)``.
        b (~chainer.Variable): Bias variable (optional) of shape ``(M,)``.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`~chainer.links.Linear`

    """
    if b is None:
        return LinearFunction()(x, W)
    else:
        return LinearFunction()(x, W, b)
