import torch.nn as nn
import torch.nn.functional as F
import torch
class D2NetLoss(nn.Module):
    """
    D2-Net loss function, composed of a L2 reconstruction loss
    and a descriptor loss.
    """
    def __init__(self, lambda_desc=250, alpha=0.9):
        super(D2NetLoss, self).__init__()

        self.reconstruction_criterion = nn.MSELoss()
        self.descriptor_criterion = nn.TripletMarginLoss(
            margin=1.0, p=2, reduction='sum'
        )
        self.lambda_desc = lambda_desc
        self.alpha = alpha

    def forward(self, output, target):
        """
        Compute D2-Net loss.

        Parameters
        ----------
        output : tuple
            Output of the network, containing the reconstructed
            image and the descriptors.
        target : tuple
            Target, containing the original image and the keypoints.

        Returns
        -------
        loss : torch.Tensor
            Total loss.
        """
        # Compute reconstruction loss
        reconstruction_loss = self.reconstruction_criterion(
            output[0], target[0]
        )

        # Compute descriptor loss
        # Positive pairs
        pos_pair_dist = F.pairwise_distance(output[1], target[1][0])
        pos_pair_loss = torch.sum(pos_pair_dist)

        # Negative pairs
        neg_pair_dist = F.pairwise_distance(output[1], target[1][1])
        neg_pair_loss = torch.sum(
            torch.clamp(self.alpha - neg_pair_dist, min=0)
        )

        descriptor_loss = (pos_pair_loss + neg_pair_loss) / \
            output[1].shape[0]

        # Total loss
        loss = reconstruction_loss + \
            self.lambda_desc * descriptor_loss

        return loss
