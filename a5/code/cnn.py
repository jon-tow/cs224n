#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# YOUR CODE HERE for part 1i


class CNN(nn.Module):
    """
    Convolutional Neural Network with ReLU activations and maxpooling for
    Character-Level Embedding.

    Named Dimensions:
    - b       = batch size
    - src_len = length of sentence
    - m_word  = maximum word length
    - e_char  = character embedding length
    - e_word  = word embedding length
    """

    def __init__(self, input_channel_count, output_channel_count, kernel_size=5):
        """
        Creates a model for combining character embeddings through a 1D
        convolution.

        Args:
          - input_channel_count: The number of channels in the input
                                 (input depth).
          - output_channel_count (int): The number of filters used in the 1D
                                 convolution. This gives the number of output
                                 channels (output depth). In our case filter
                                 will be embed_word_size (e_word)
          - kernel_size (int): The size of the window used to compute features.
                               Default: 5
        """
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_channel_count,
                              out_channels=output_channel_count,
                              kernel_size=kernel_size)
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, input):
        """
        The forward pass of the Character-Level CNN that maps the reshaped
        input to the convolution based embedding output embeddings.

        input -> conv -> relu -> maxpool

        Args:
          - input (torch.Tensor): The input to the layer. This will be the
                                  reshaped character embeddings from
                                  the preceding embedding layer of shape:
            (src_len * b, e_char, m_word)

        Returns:
          - output (torch.Tensor): The highway output word embeddings of shape
                                   (src_len * b, e_word).
        """
        # Tensor: (src_len * b, e_word, m_word - kernel_size + 1)
        activations = F.relu(self.conv(input))
        # Tensor: (src_len * b, e_word)
        output = self.maxpool(activations).squeeze(dim=2)
        return output

# END YOUR CODE
