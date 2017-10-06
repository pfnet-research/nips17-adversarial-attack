import chainer.functions as F
import chainer.initializers as I
import chainer.links as L

from .tf_loadable_chain import TFLoadableChain


class ConvBnRelu(TFLoadableChain):

    def __init__(self, depth, ksize, stride=1, pad=0, initialW=I.HeNormal()):
        super(ConvBnRelu, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, depth, ksize=ksize, stride=stride, pad=pad, initialW=initialW, nobias=True)
            self.bn = L.BatchNormalization(depth, decay=0.9997, eps=0.001, use_gamma=False)

    def __call__(self, x):
        return F.relu(self.bn(self.conv(x)))

    def load_tf_checkpoint(self, reader, name):
        try:
            self.conv.W.data[...] = reader.get_tensor(name + '/weights').transpose(3, 2, 0, 1)
            self.bn.beta.data[...] = reader.get_tensor(name + '/BatchNorm/beta')
            self.bn.avg_mean[...] = reader.get_tensor(name + '/BatchNorm/moving_mean')
            self.bn.avg_var[...] = reader.get_tensor(name + '/BatchNorm/moving_variance')
        except Exception:
            print('failed at', name)
            raise
