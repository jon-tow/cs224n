#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# YOUR CODE HERE for part 1h


class Highway(nn.Module):
    """
    Highway Network Network for Character-Level Embedding.

    Named Dimensions:
    - b       = batch size
    - src_len = length of sentence
    - e_word  = word embedding length
    """

    def __init__(self, input_size):
        """
        Creates a highway network to improve the word embeddings from a
        Character-Level Convolutional Neural Network.

        Args:
          - input_size (int): The size of the feature dimension of the incoming
                        tensor.
                        Note: the input and output dimensions will be the same.
        """
        super(Highway, self).__init__()
        self.gate = nn.Linear(in_features=input_size,
                              out_features=input_size)
        self.proj = nn.Linear(in_features=input_size,
                              out_features=input_size)

    def forward(self, input):
        """
        The forward pass of a Highway block that maps from the convolution
        output of the previous layer to the highway's output.

        Args:
          - input (torch.Tensor): The input to the layer. This will be the
                                  output of the convolution layer, i.e.
                                  the preceding layer, with shape:
                                  (src_len * b, e_word).
        Returns:
          - output (torch.Tensor): The highway output of shape
                                   (src_len * b, e_word).
        """
        proj = F.relu(self.proj(input))
        gate = torch.sigmoid(self.gate(input))
        highway = gate * proj + (1 - gate) * input
        return highway

# END YOUR CODE
