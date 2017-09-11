from chainer import function
from chainer.utils import type_check
from time import time

import micpy

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

    def forward(self, inputs):
        x = _as_mat(inputs[0])
        end = time() - start
        W = inputs[1]
        start = time()
        y = x.dot(W.T).astype(x.dtype, copy=False)
        end = time() - start
        with open("/home/minhtri/workspace/numpy_test/workspace/log/log6.txt","a") as file_log:
            file_log.write("dot operate on y of linear function time(forward): {} \n".format(end))
        # with open("/home/minhtri/workspace/numpy_test/workspace/log/log_type.txt","a") as file_log:
        #     file_log.write("(Forward dot) x data type: {} \n".format(type(x)))
        #     file_log.write("(Forward dot) x shape: {} \n".format(x.shape)) 
        #     file_log.write("(Forward dot) x dtype: {} \n".format(x.dtype))
        #     file_log.write("(Forward dot) W data type: {} \n".format(type(W)))
        #     file_log.write("(Forward dot) W shape: {} \n".format(W.shape))
        #     file_log.write("(Forward dot) W dtype: {} \n".format(W.dtype)) 
        if len(inputs) == 3:
            b = inputs[2]
            with open("/home/minhtri/workspace/numpy_test/workspace/log/log7.txt","a") as file_log:
                file_log.write("+= is start \n")
            start = time()
            y += b 
            end = time() - start
            with open("/home/minhtri/workspace/numpy_test/workspace/log/log7.txt","a") as file_log:
                file_log.write("+= is end \n")
            with open("/home/minhtri/workspace/numpy_test/workspace/log/log6.txt","a") as file_log:
                file_log.write("add operate on y of linear function time(forward): {} \n".format(end))
            with open("/home/minhtri/workspace/numpy_test/workspace/log/log_type.txt","a") as file_log:
                file_log.write("(Forward iadd) y data type: {} \n".format(type(y)))
                file_log.write("(Forward iadd) y shape: {} \n".format(y.shape)) 
                file_log.write("(Forward iadd) y dtype: {} \n".format(y.dtype))
                file_log.write("(Forward iadd) b data type: {} \n".format(type(b)))
                file_log.write("(Forward iadd) b shape: {} \n".format(b.shape))
                file_log.write("(Forward iadd) b dtype: {} \n".format(b.dtype)) 
        return y,

    def backward(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        W = inputs[1]
        gy = grad_outputs[0]
        start = time()
        gx = gy.dot(W).astype(x.dtype, copy=False).reshape(inputs[0].shape)
        end = time() - start
        with open("/home/minhtri/workspace/numpy_test/workspace/log/log6.txt","a") as file_log:
            file_log.write("dot operate on gx of linear function time(backward): {} \n".format(end))
        start = time()
        gW = gy.T.dot(x).astype(W.dtype, copy=False)
        end = time() - start
        with open("/home/minhtri/workspace/numpy_test/workspace/log/log6.txt","a") as file_log:
            file_log.write("dot operate on gW of linear function time(backward): {} \n".format(end))
        with open("/home/minhtri/workspace/numpy_test/workspace/log/log_type.txt","a") as file_log:
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
            start = time()
            gb = gy.sum(0)
            end = time() - start
            with open("/home/minhtri/workspace/numpy_test/workspace/log/log6.txt","a") as file_log:
                file_log.write("sum operate on gb of linear function time(backward): {} \n".format(end))
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
