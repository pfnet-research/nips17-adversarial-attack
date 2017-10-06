import chainer
import glob
import os
import numpy as np
import chainer.configuration
import cv2

import datasets
import multi_fcn_training
import math


def normalize_maxp(x):
    x = float(x)
    x = math.floor(x)
    x = min(16, x)
    x = max(4, x)
    return x


def main(
        # Dataset
        input_dir='/mnt/sakuradata10-striped/nips17_adversarial/dataset/images',
        output_dir='./out',
        max_perturbation=16,
        # Model
        model_file=None,
        resnet50_file=None,
        incresv2_file=None,
        # Devices
        gpu=0,
        batchsize=4,
        loaderjob=4,
        # Configuration
        n_iters=1,
):
    chainer.cuda.get_device(gpu).use()

    image_abs_paths = glob.glob(os.path.join(input_dir, '*.png'))
    image_rel_paths = [os.path.basename(path) for path in image_abs_paths]

    image_dataset = datasets.TestImageDataset(image_abs_paths, 299)
    iterator = chainer.iterators.SerialIterator(
        image_dataset, batchsize, repeat=False, shuffle=False)

    import fcn_models.rec_multibp_resnet
    fcn = fcn_models.rec_multibp_resnet.RecMultiBPResNet(
        n_out=3*13, resnet50_path=resnet50_file, incresv2_path=incresv2_file)

    print('Model:{}, File: {}, MaxP: {}'.format(type(fcn), model_file, max_perturbation))
    chainer.serializers.load_npz(model_file, fcn)
    fcn.to_gpu()

    print('Images: ', len(image_dataset))

    #
    # Predict
    #
    chainer.configuration.config.max_perturbation = max_perturbation
    chainer.configuration.config.n_iters = n_iters

    with chainer.configuration.using_config('train', False):
        with chainer.function.no_backprop_mode():
            i = 0

            iterator.reset()
            for batch in iterator:
                x_299 = chainer.dataset.convert.concat_examples(batch, gpu)
                x_299 = multi_fcn_training.perturbate_non_targeted(x_299, fcn, max_perturbation)

                for img in x_299:  # CHW, BGR
                    img = chainer.cuda.to_cpu(img.data)
                    output_path = os.path.join(output_dir, image_rel_paths[i])
                    i += 1

                    img = img.transpose(1, 2, 0)  # HWC
                    img = np.rint(img).clip(0, 255).astype(np.uint8)
                    cv2.imwrite(output_path, img)
                    print(i, output_path)

            assert(i == len(image_dataset))


if __name__ == '__main__':
    import fire
    fire.Fire(main)
