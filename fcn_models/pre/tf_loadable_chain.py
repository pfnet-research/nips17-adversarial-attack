import chainer
import chainer.links as L


class TFLoadableChain(chainer.Chain):

    """Chain class that support load_tf_checkpoint() method."""

    def __init__(self, final_layer_name=''):
        super(TFLoadableChain, self).__init__()
        self.final_layer_name = final_layer_name

    def load_tf_checkpoint(self, reader, path):
        """Load TF checkpoint.

        It calls ``load_tf_checkpoint()`` of the direct children, or extract the weights and biases for ``Linear`` and
        ``Convolution2D`` chains. The final layer of the network is specially treated; the original models output 1,001
        way label scores including "background" label, while our models writtein in Chainer output 1,000 way label
        scores. The name of the final layer is specified by the ``final_layer_name`` attribute.

        Args:
            reader (CheckpointReader): Checkpoint reader. It can be created by
                ``tensorflow.python.pywrap_tensorflow.NewCheckpointReader(path)``.
            path (str): Root object path.

        """
        for child in self.children():
            full_name = '{}/{}'.format(path, child.name)
            if isinstance(child, (L.Convolution2D, L.Linear)):
                try:
                    start_index = int(child.name == self.final_layer_name)
                    W = reader.get_tensor(full_name + '/weights')
                    if W.ndim == 4:  # conv2d
                        W = W.transpose(3, 2, 0, 1)
                    else:  # linear
                        W = W.T
                    child.W.data[...] = W[start_index:]
                    if hasattr(child, 'b'):
                        b = reader.get_tensor(full_name + '/biases')
                        child.b.data[...] = b[start_index:]
                except Exception:
                    print('failed at', full_name)
                    raise
            else:
                child.load_tf_checkpoint(reader, full_name)


class TFLoadableRepeat(chainer.ChainList):

    """ChainList that support load_tf_checkpoint() method.

    It corresponds to ``slim.repeat()``.

    """
    def __init__(self, generator, count, genname):
        super(TFLoadableRepeat, self).__init__()
        for i in range(count):
            self.add_link(generator())
        self.genname = genname

    def __call__(self, x):
        for layer in self:
            x = layer(x)
        return x

    def load_tf_checkpoint(self, reader, path):
        for i, child in enumerate(self, 1):
            child.load_tf_checkpoint(reader, '{}/{}_{}'.format(path, self.genname, i))
