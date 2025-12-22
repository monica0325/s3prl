# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ add_noise.py ]
#   Synopsis     [ Class for adding random Gaussian noise to features ]
#   Author       [ Your Name ]
#   Description  [ Adds random noise to audio feature tensors to simulate augmentation ]
"""*********************************************************************************************"""

import torch

class AddNoise(torch.nn.Module):
    """Class to add random Gaussian noise to features."""
    def __init__(self, location="wav", noise_mean=0.0, noise_std=0.005, intensity=1.0):
        """
        Args:
            noise_mean (float): Mean of the Gaussian noise.
            noise_std (float): Standard deviation of the Gaussian noise.
            intensity (float): Multiplier for noise_std to control noise intensity.
        """
        super(AddNoise, self).__init__()
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.intensity = intensity
        self.location = location

    def add_noise(self, x):
        """Applies random Gaussian noise to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, features).
        
        Returns:
            torch.Tensor: Tensor with added Gaussian noise.
        """
        # Scale the standard deviation by the intensity
        adjusted_std = self.noise_std * self.intensity
        noise = torch.normal(self.noise_mean, adjusted_std, size=x.size(), device=x.device)
        return x + noise

    def forward(self, xs, x_lengths=None):
        """
        Args:
            xs (list[torch.Tensor]): List of feature tensors [(T, D)] x batchsize.
            x_lengths (torch.Tensor, optional): Length of each feature sequence in the batch.
        
        Returns:
            list[torch.Tensor]: List of features with added noise [(T, D)] x batchsize.
        """
        assert len(xs[0].size()) == 2  # Ensure input is a list of 2D tensors (T, D)

        # Determine padding for batch
        if x_lengths is None:
            x_lengths = torch.LongTensor([x.size(0) for x in xs])
        batchsize, max_len, dim = len(x_lengths), torch.max(x_lengths).item(), xs[0].size(1)

        # Pad sequences to the same length
        xs_pad = xs[0].new_zeros((batchsize, max_len, dim))
        for i, x in enumerate(xs):
            xs_pad[i, :x_lengths[i]] = x

        # Add random Gaussian noise
        xs_pad = self.add_noise(xs_pad)

        # Restore individual sequences
        xs_noisy = [xs_pad[i, :xs[i].size(0)] for i in range(batchsize)]
        return xs_noisy, x_lengths # list of augmented features