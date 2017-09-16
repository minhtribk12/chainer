from chainer import function
from chainer.utils import type_check
from time import time

import micpy
import pymic as mic
import numpy as np

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
        m = operand1.shape[0]
        n = operand1.shape[1]
        c_ = operand1
        a_ = numpy.tile(operand2,(m,1))
        device_mic = pymic.devices[0]
        library_mic = device_mic.load_library("libdgemm.so")
        stream_mic = device_mic.get_default_stream()
        alpha_mic = 1.0
        beta_mic = 1.0
        b_ = numpy.diag(numpy.ones(n)).reshape((n,n))
        offl_a = stream_mic.bind(a_)
        offl_b = stream_mic.bind(b_)
        offl_c = stream_mic.bind(c_)
        stream_mic.invoke(library_mic.dgemm_kernel, offl_a, offl_b, offl_c, m, n, n, alpha_mic, beta_mic)
        stream_mic.sync()
        offl_c.update_host()
        stream_mic.sync()
        return offl_c.array
    # def dot_mic(self, operand1, operand2):
    #     m = operand1.shape[0]
    #     k = operand1.shape[1]
    #     n = operand2.shape[1]
    #     a_ = operand1
    #     b_ = operand2
    #     c_ = numpy.zeros((m,n))
    #     device_mic = pymic.devices[0]
    #     library_mic = device_mic.load_library("libdgemm.so")
    #     with open("./log/log7.txt","a") as file_log: 
    #         file_log.write("point 1 \n")
    #     stream_mic = device_mic.get_default_stream()
    #     with open("./log/log7.txt","a") as file_log: 
    #         file_log.write("point 2 \n")
    #     alpha_mic = 1.0
    #     beta_mic = 0.0
    #     with open("./log/log7.txt","a") as file_log: 
    #         file_log.write("point 3 \n")
    #     offl_a = stream_mic.bind(a_)
    #     with open("./log/log7.txt","a") as file_log: 
    #         file_log.write("point 4 \n")
    #     offl_b = stream_mic.bind(b_)
    #     with open("./log/log7.txt","a") as file_log: 
    #         file_log.write("point 5 \n")
    #     offl_c = stream_mic.bind(c_)
    #     with open("./log/log7.txt","a") as file_log: 
    #         file_log.write("invoke start \n")
    #     stream_mic.invoke(library_mic.dgemm_kernel, offl_a, offl_b, offl_c, m, n, k, alpha_mic, beta_mic)
    #     with open("./log/log7.txt","a") as file_log: 
    #         file_log.write("invoke end \n")
    #     stream_mic.sync()
    #     offl_c.update_host()
    #     stream_mic.sync()
    #     return offl_c.array
    def dot_mic(self, operand1, operand2):
        a = operand1.astype(np.float32)
        b = operand2.astype(np.float32)
        device = mic.devices[0]
        library = device.load_library("libtests.so")
        stream = device.get_default_stream()
        m, n, k = a.shape[0], b.shape[1], a.shape[1]
        alpha, beta = 1.0, 0
        c = np.zeros((m, n))
        offl_a = stream.bind(a)
        offl_b = stream.bind(b)
        offl_c = stream.bind(c)
        stream.sync()
        with open("./log/log7.txt","a") as file_log: 
             file_log.write("point 1 \n")
        stream.invoke(library.sgemm_kernel,
                      offl_a, offl_b, offl_c,
                      m, n, k, alpha, beta)
        with open("./log/log7.txt","a") as file_log: 
             file_log.write("point 2 \n")
        #stream.sync()
        offl_c.update_host()
        stream.sync()
        with open("./log/log7.txt","a") as file_log: 
             file_log.write("point 3 \n")
        #r = c.copy()
        return c
    #End
    def forward(self, inputs):
        with open("./log/log7.txt","a") as file_log: 
            file_log.write("forward linear start \n")
        x = _as_mat(inputs[0])
        W = inputs[1]
        with open("./log/log7.txt","a") as file_log: 
            file_log.write("dot start \n")
        with open("./log/log_type.txt","a") as file_log:
            file_log.write("(Forward iadd) x data type: {} \n".format(type(x)))
            file_log.write("(Forward iadd) x shape: {} \n".format(x.shape)) 
            file_log.write("(Forward iadd) x dtype: {} \n".format(x.dtype))
            file_log.write("(Forward iadd) W data type: {} \n".format(type(W)))
            file_log.write("(Forward iadd) W shape: {} \n".format(W.shape))
            file_log.write("(Forward iadd) W dtype: {} \n".format(W.dtype))
        start = time()
        #y = x.dot(W.T).astype(x.dtype, copy=False)
        y = self.dot_mic(x,(W.T)).astype(x.dtype)
        #y = u.copy()
        #del u
        end = time() - start
        with open("./log/log6.txt","a") as file_log:
            file_log.write("dot operate on y of linear function time(forward): {} \n".format(end))
        with open("./log/log7.txt","a") as file_log: 
            file_log.write("dot end \n")
        # with open("./log/log_type.txt","a") as file_log:
        #     file_log.write("(Forward dot) x data type: {} \n".format(type(x)))
        #     file_log.write("(Forward dot) x shape: {} \n".format(x.shape)) 
        #     file_log.write("(Forward dot) x dtype: {} \n".format(x.dtype))
        #     file_log.write("(Forward dot) W data type: {} \n".format(type(W)))
        #     file_log.write("(Forward dot) W shape: {} \n".format(W.shape))
        #     file_log.write("(Forward dot) W dtype: {} \n".format(W.dtype)) 
        if len(inputs) == 3:
            b = inputs[2]
            with open("./log/log7.txt","a") as file_log:
                file_log.write("+= start \n")
            start = time()
            y += b
            #y = self.iadd_mic(y, b)
            end = time() - start
            with open("./log/log7.txt","a") as file_log:
                file_log.write("+= is end \n")
            with open("./log/log6.txt","a") as file_log:
                file_log.write("+= operate on y of linear function time(forward): {} \n".format(end))
            with open("./log/log_type.txt","a") as file_log:
                file_log.write("(Forward iadd) y data type: {} \n".format(type(y)))
                file_log.write("(Forward iadd) y shape: {} \n".format(y.shape)) 
                file_log.write("(Forward iadd) y dtype: {} \n".format(y.dtype))
                file_log.write("(Forward iadd) b data type: {} \n".format(type(b)))
                file_log.write("(Forward iadd) b shape: {} \n".format(b.shape))
                file_log.write("(Forward iadd) b dtype: {} \n".format(b.dtype))
        with open("./log/log7.txt","a") as file_log: 
            file_log.write("forward linear end \n") 
        return y,

    def backward(self, inputs, grad_outputs):
        with open("./log/log7.txt","a") as file_log: 
            file_log.write("backward linear start \n")
        x = _as_mat(inputs[0])
        W = inputs[1]
        gy = grad_outputs[0]
        with open("./log/log7.txt","a") as file_log: 
            file_log.write("dot start \n")
        start = time()
        #gx = gy.dot(W).astype(x.dtype, copy=False).reshape(inputs[0].shape)
        gx = self.dot_mic(gy,W).astype(x.dtype).reshape(inputs[0].shape)
        end = time() - start
        with open("./log/log6.txt","a") as file_log:
            file_log.write("dot operate on gx of linear function time(backward): {} \n".format(end))
        with open("./log/log7.txt","a") as file_log: 
            file_log.write("dot end \n")
        with open("./log/log7.txt","a") as file_log: 
            file_log.write("dot 2 start \n")
        start = time()
        #gW = gy.T.dot(x).astype(W.dtype, copy=False)
        gW = self.dot_mic((gy.T),x).astype(W.dtype)
        end = time() - start
        with open("./log/log6.txt","a") as file_log:
            file_log.write("dot operate on gW of linear function time(backward): {} \n".format(end))
        with open("./log/log7.txt","a") as file_log: 
            file_log.write("dot 2 end \n")
        with open("./log/log_type.txt","a") as file_log:
            file_log.write("(Backward) gy data type: {} \n".format(type(gy))) 
            file_log.write("(Backward) gy shape: {} \n".format(gy.shape)) 
            file_log.write("(Backward) gy dtype: {} \n".format(gy.dtype))
            file_log.write("(Backward) x data type: {} \n".format(type(x)))
            file_log.write("(Backward) x shape: {} \n".format(x.shape)) 
            file_log.write("(Backward) x dtype: {} \n".format(x.dtype))
            file_log.write("(Backward) W data type: {} \n".format(type(W)))
            file_log.write("(Backward) W shape: {} \n".format(W.shape))
            file_log.write("(Backward) W dtype: {} \n".format(W.dtype)) 
        if len(inputs) == 3:
            with open("./log/log7.txt","a") as file_log: 
                file_log.write("sum start \n")
            start = time()
            gb = gy.sum(0)
            end = time() - start
            with open("./log/log6.txt","a") as file_log:
                file_log.write("sum operate on gb of linear function time(backward): {} \n".format(end))
            with open("./log/log7.txt","a") as file_log: 
                file_log.write("sum end \n")
            with open("./log/log7.txt","a") as file_log: 
                file_log.write("backward linear end \n")
            return gx, gW, gb
        else:
            with open("./log/log7.txt","a") as file_log: 
                file_log.write("backward linear end \n")
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
