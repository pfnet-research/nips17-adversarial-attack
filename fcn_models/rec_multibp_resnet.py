import chainer
import chainer.functions as F
import chainer.links as L
import random

from fcn_models.resnet_layer import ResNet50Layers
from .pre.inception_resnet_v2 import InceptionResNetV2Layers


class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch):
        super(BottleNeckB, self).__init__()
        initialW = chainer.initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(
                ch, in_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(in_size, initial_gamma=0)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))

        return F.relu(h + x)


class UpBlock(chainer.Chain):

    def __init__(self, in_ch_dec, in_ch_enc, out_ch):
        super(UpBlock, self).__init__()

        with self.init_scope():
            self.d0 = L.Deconvolution2D(in_ch_dec, out_ch, 2, 2)
            self.b0 = L.BatchNormalization(out_ch)

            self.c1 = L.Convolution2D(out_ch + in_ch_enc, out_ch, 3, 1, 1)
            self.b1 = L.BatchNormalization(out_ch)

            self.c2 = L.Convolution2D(out_ch, out_ch, 3, 1, 1)
            self.b2 = L.BatchNormalization(out_ch)

    def __call__(self, h, *h_skip):
        h = F.relu(self.b0(self.d0(h)))
        h = F.concat((h, *h_skip))
        h = F.relu(self.b1(self.c1(h)))
        h = F.relu(self.b2(self.c2(h)))

        return h


class Mix(chainer.Chain):

    def __init__(self):
        super(Mix, self).__init__()

        enc_ch = [3, 64, 256, 512, 1024, 2048]
        ins_ch = [6, 128, 384, 640, 2176, 3072]

        self.conv = [None] * 6
        self.bn = [None] * 6
        for i in range(1, 6):
            c = L.Convolution2D(enc_ch[i] + ins_ch[i], enc_ch[i], 1, nobias=True)
            b = L.BatchNormalization(enc_ch[i])

            self.conv[i] = c
            self.bn[i] = b

            self.add_link('c{}'.format(i), c)
            self.add_link('b{}'.format(i), b)

    def __call__(self, ss):
        ss = list(ss)
        for i in range(1, 6):
            ss[i] = F.relu(self.bn[i](self.conv[i](ss[i])))
        return ss


class Decoder(chainer.Chain):

    def __init__(self, out_ch):
        super(Decoder, self).__init__()

        with self.init_scope():
            self.mix = Mix()

            self.bot1 = BottleNeckB(2048, 1024)
            self.bot2 = BottleNeckB(2048, 1024)
            self.bot3 = BottleNeckB(2048, 1024)

            self.b5 = UpBlock(2048, 1024, 1024)
            self.b4 = UpBlock(1024, 512, 512)
            self.b3 = UpBlock(512, 256, 256)
            self.b2 = UpBlock(256, 64, 128)
            self.b1 = UpBlock(128, 3 + (6 + 3 * 13), 64)

            self.last_b = L.BatchNormalization(64)
            self.last_c = L.Convolution2D(64, out_ch * 2, 1, nobias=True)

    def __call__(self, x, p0, ss):
        #
        # State mixer
        #
        ss = self.mix(ss)

        #
        # Bottleneck
        #
        h = ss[5]
        h = self.bot1(h)
        h = self.bot2(h)
        h = self.bot3(h)

        #
        # Up blocks
        #
        h = self.b5(h, ss[4])
        h = self.b4(h, ss[3])
        h = self.b3(h, ss[2])
        h = self.b2(h, ss[1])
        h = self.b1(h, x, ss[0])

        #
        # Last
        #
        h = self.last_c(F.relu(self.last_b(h)))
        p1, q = F.split_axis(h, 2, axis=1)
        p1 = F.tanh(p1)
        q = F.sigmoid(q)
        ss[0] = p1 * q + p0 * (1.0 - q)

        return tuple(ss)


class RNN(chainer.Chain):

    def __init__(self, n_out, incresv2_path):
        super(RNN, self).__init__()
        self.ins = InceptionResNetV2Layers(incresv2_path)

        with self.init_scope():
            self.dec = Decoder(n_out)

    def __call__(self, x, ss):
        # Inception (we don't need forget, as we don't backprop)
        p0 = ss[0]
        c = (chainer.configuration.config.max_perturbation - 4) * 3
        x_plus_p0 = x + p0[:, c:c + 3, :, :]
        ds = self.ins(x_plus_p0)
        ss = [F.concat((s, d), axis=1) for s, d in zip(ss, ds)]

        # Decode (forget)
        assert isinstance(x, chainer.Variable)
        ss = F.forget(lambda x_, p0_, *ss_: self.dec(x_, p0_, ss_), x, p0, *ss)

        return ss


class Encoder(chainer.Chain):

    def __init__(self, resnet50_path):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.enc = ResNet50Layers(pretrained_model=resnet50_path)

    def __call__(self, x):
        enc_layers = ["conv1", "res2", "res3", "res4", "res5"]
        hs_enc = self.enc(x, layers=enc_layers)
        return tuple([hs_enc[layer] for layer in enc_layers])


MAX_N_ITERS = 2


class RecMultiBPResNet(chainer.Chain):

    def __init__(self, n_out, resnet50_path='auto', incresv2_path=None):
        super(RecMultiBPResNet, self).__init__()

        self.n_out = n_out
        self.i = 0

        with self.init_scope():
            self.enc = Encoder(resnet50_path)
            self.rnn = RNN(n_out, incresv2_path)

    def n_iters(self):
        if chainer.configuration.config.train:
            n_iters = 1 + (self.i % MAX_N_ITERS)  # random.randint(1, MAX_N_ITERS)
            self.i += 1
        else:
            if chainer.configuration.config.n_iters is not None:
                n_iters = chainer.configuration.config.n_iters
            else:
                n_iters = MAX_N_ITERS
        return n_iters

    def __call__(self, x):
        xp = chainer.cuda.get_array_module(x, x.data)
        p0 = xp.zeros(x.shape[0:1] + (self.n_out, ) + x.shape[2:], dtype=xp.float32)
        p0 = chainer.Variable(p0)

        # Encode (forget)
        assert isinstance(x, chainer.Variable)
        ss = [p0] + list(F.forget(lambda x_: self.enc(x_), x))

        # RNN
        n_iters = self.n_iters()
        for i in range(n_iters):
            ss = self.rnn(x, ss)

        return ss[0]
