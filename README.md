# ProjUNN: efficient learning of orhtogonal or unitary weights by low-rank updates


## Convolutional orthogonality/unitarity constraints

To run the convolution case simply run

> python convolutional_experiment.py --unitary --dataset MNIST -lr 0.001 --projector projUNNT --optimizer SGD

with the desired settings. the projector option can be either `projUNND` or `projUNNT` for the two methods we proposed in the paper. The optimizer is either `SGD` or `RMSProp`. Note that the otpimizers (other than simple SGD) need to be rewritten to be sure that terms such as momentum etc. are computed on the gradients/projected gradients but that the update to the weights is projected (to ensure that the weights stay on the orthogonal/unitary manifolds). As of now, this code simply runs the Resnet9 model on `MNIST`, `CIFAR10`, and `CIFAR100`.

To run without the unitary constraint simply remove the `--unitary` flag.

## Requirements

This software only requires `pytorch` and all its dependencies.

## TERMS OF USE & PRIVACY POLICY
- Terms of Use - https://opensource.facebook.com/legal/terms
- Privacy Policy - https://opensource.facebook.com/legal/privacy

## COPYRIGHT STATEMENT
Copyright © 2022 Meta Platforms, Inc

## LICENSE

MIT
