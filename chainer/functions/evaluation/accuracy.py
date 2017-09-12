import numpy
import six

from chainer import cuda
from chainer import mic
from chainer import function
from chainer.utils import type_check
from time import time


class Accuracy(function.Function):

    def __init__(self, ignore_label=None):
        self.ignore_label = ignore_label

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype == numpy.int32
        )

        t_ndim = t_type.ndim.eval()
        type_check.expect(
            x_type.ndim >= t_type.ndim,
            x_type.shape[0] == t_type.shape[0],
            x_type.shape[2: t_ndim + 1] == t_type.shape[1:]
        )
        for i in six.moves.range(t_ndim + 1, x_type.ndim.eval()):
            type_check.expect(x_type.shape[i] == 1)

    def forward(self, inputs):
        if any(isinstance(i, mic.ndarray) for i in inputs):
            return self.forward_mic(inputs)

        xp = cuda.get_array_module(*inputs)
        y, t = inputs

        if self.ignore_label is not None:
            mask = (t == self.ignore_label)
            ignore_cnt = mask.sum()

            # will always be true when the true label is ignore_label
            # TODO(henry0312)
            #   If cupy.where returns indexes, we could make the code better.
            #   Also, we would need Advanced Indexing.
            pred = xp.where(mask, self.ignore_label,
                            y.argmax(axis=1).reshape(t.shape))
            count = (pred == t).sum() - ignore_cnt
            total = t.size - ignore_cnt

            if total == 0:
                return xp.asarray(0.0, dtype=y.dtype),
            else:
                return xp.asarray(float(count) / total, dtype=y.dtype),
        else:
            pred = y.argmax(axis=1).reshape(t.shape)
            return xp.asarray((pred == t).mean(dtype=y.dtype)),

    def forward_mic(self, inputs):
        #Cause this function is non-differentiable and do not play any
        #roles in backward stage, these output will be numpy array
        micpy = mic.micpy
        y, t = inputs

        if self.ignore_label is not None:
            mask = (t == self.ignore_label)
            start = time()
            ignore_cnt = mask.sum()
            end = time() - start
            with open("./log/log6.txt","a") as file_log:
                file_log.write("sum operate on ignore_cnt in accuracy function time: {}\n".format(end))
            start = time()
            pred = y.argmax(axis=1).reshape(t.shape)
            end = time() - start
            with open("./log/log6.txt","a") as file_log:
                file_log.write("argmax & reshape operate 1 in accuracy function time: {}\n".format(end))
            start = time()
            count = ((pred == t) | mask).sum() - ignore_cnt
            end = time() - start
            with open("./log/log6.txt","a") as file_log:
                file_log.write("sum operate on count in accuracy function time: {}\n".format(end))
            total = t.size - ignore_cnt

            if total == 0:
                return numpy.asarray(0.0, dtype=y.dtype)
            else:
                return numpy.asarray(float(count) / total, dtype=y.dtype)
        else:
            start = time()
            pred = y.argmax(axis=1).reshape(t.shape)
            end = time() - start
            with open("./log/log6.txt","a") as file_log:
                file_log.write("argmax & reshape operate 2 in accuracy function time: {}\n".format(end))
            start = time()
            count = (pred == t).sum()
            end = time() - start
            with open("./log/log6.txt","a") as file_log:
                file_log.write("sum operate on count 2 in accuracy function time: {}\n".format(end))
            return numpy.asarray(float(count) / t.size, dtype=y.dtype),


def accuracy(y, t, ignore_label=None):
    """Computes muticlass classification accuracy of the minibatch.

    Args:
        y (Variable): Variable holding a matrix whose (i, j)-th element
            indicates the score of the class j at the i-th example.
        t (Variable): Variable holding an int32 vector of ground truth labels.
        ignore_label (int or None): Skip calculating accuracy
            if the true label is ``ignore_label``.

    Returns:
        Variable: A variable holding a scalar array of the accuracy.

    .. note:: This function is non-differentiable.

    """
    return Accuracy(ignore_label=ignore_label)(y, t)
