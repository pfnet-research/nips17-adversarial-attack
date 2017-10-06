import chainer
import random


def pad_299_to_320(x_299):
    x_320 = chainer.functions.pad(x_299, ((0, 0), (0, 0), (10, 11), (10, 11)), 'constant', constant_values=128)
    return x_320


def perturbate_non_targeted(x_299, fcn, magnitude):
    assert (isinstance(magnitude, int) and 4 <= magnitude <= 16)

    x_320 = pad_299_to_320(x_299)
    p_320 = fcn(x_320)
    del x_320

    c = (magnitude - 4) * 3
    p_299 = p_320[:, c:c + 3, 10:309, 10:309]

    st = p_299.data.std()
    ma = max(-p_299.data.min(), p_299.data.max())
    chainer.reporter.report({'std': st, 'max': ma})

    p_299 = p_299 * float(magnitude)

    x_299 = p_299 + x_299
    return x_299
