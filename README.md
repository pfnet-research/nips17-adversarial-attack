# Kaggle NIPS'17 Competition

This repository contains the submission of team 'iwiwi' for the non-targeted adversarial attack track of Kaggle NIPS'17 competition on adversarial examples (https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack).

## Overview

Our approach is to produce adversarial examples by using fully-convolutional neural networks. The basic framework is the same as that of the Adversarial Transformation Networks paper (https://arxiv.org/pdf/1703.09387.pdf), but we used a much larger FCN model and stronger computation power, together with several new ideas such as multi-target training, multi-task training, and gradient hints. For details, we are preparing a technical report that describes our approach.


## How to Run

* Download our attacker model:
	* Download from https://www.dropbox.com/s/x5kcq8gjil4kj2p/000030-1001-1416-resume3e-4-5000_model_multi_iter2000.npz?dl=0
	* Place it as `000030-1001-1416-resume3e-4-5000_model_multi_iter2000.npz`
* Download Inception ResNet v2 model with ensemble adversarial training (which is used for gradient hints):
	* Download from http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz 
	* Place it as `ens_adv_inception_resnet_v2.ckpt.{index, meta, data-00000-of-00001}`
* Run the following commands.

```
docker pull iwiwi/nips17-adversarial
nvidia-docker run \
  -v ${INPUT_IMAGES}:/input_images \
  -v ${OUTPUT_IMAGES}:/output_images \
  -v ${SUBMISSION_DIRECTORY}:/code \
  -w /code \
  iwiwi/nips17-adversarial \
  ./run_attack.sh \
  /input_images \
  /output_images \
  ${MAX_PERTURBATION}
```


## Examples

The following is the examples of our attack with MAX_PERTURBATION=16 (left: original image, middle: perturbated image, right: perturbation).

![Example images](examples.png)


## References

* Shumeet Baluja, Ian Fischer. **Adversarial Transformation Networks: Learning to Generate Adversarial Examples.** *CoRR*, abs/1703.09387, 2017.


## License

[MIT License](LICENSE)
