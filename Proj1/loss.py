

import torch
import torch.nn.functional as F


def custom_loss(output, target, classes, lambda_=1.0):
    """
    Custom loss for network with auxiliary losses. The total loss is a combination
    of the loss of the main task (binary cross entropy) and the negative log likelihood
    for the two auxiliary tasks. Importance of auxiliary losses is controlled by
    the `lambda_` hyperparameter.
    """
    main, im1, im2 = output

    main_loss = F.binary_cross_entropy(main.flatten(), target)
    aux_loss_1 = F.nll_loss(im1, classes[:, 0])
    aux_loss_2 = F.nll_loss(im2, classes[:, 1])

    return main_loss + lambda_ * (aux_loss_1 + aux_loss_2)


def handle_loss(criterion):
    """
    Handle the fact that the network with auxiliary loss has three-item tuple output,
    which needs to be treated separately to compute the loss and the accuracy.
    """
    if criterion is not custom_loss:
        internal_criterion = lambda output, target, _: criterion(output.flatten(), target)
        compute_acc = lambda output, target: (target == torch.round(output.flatten())).sum().item()
    else:
        internal_criterion = criterion
        compute_acc = lambda output, target: (target == torch.round(output[0].flatten())).sum().item()

    return internal_criterion, compute_acc
