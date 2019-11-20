#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change"


class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output.
                                 Notation: e_word.
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for
                                   documentation.
        """
        super(ModelEmbeddings, self).__init__()

        # A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(
        #     len(vocab.src), embed_size, padding_idx=pad_token_idx)
        # End A4 code

        # YOUR CODE HERE for part 1j

        pad_token_idx = vocab.char2id['<pad>']
        self.embed_char_size = 50  # e_char = 50 (see 1. (j) bullet point 5)
        self.embed_size = embed_size  # e_word

        self.char_embed = nn.Embedding(num_embeddings=vocab.char_count(),
                                       embedding_dim=self.embed_char_size,
                                       padding_idx=pad_token_idx)
        self.cnn = CNN(input_channel_count=self.embed_char_size,
                       output_channel_count=self.embed_size)
        self.highway = Highway(input_size=self.embed_size)
        self.dropout = nn.Dropout(p=0.3)

        # END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of
        sentences.

        @param input: Tensor of integers of shape
            (src_len = sentence_length, b = batch_size, m_word = max_word_length)
            where each integer is an index into the character vocabulary.

        @returns output: Tensor of floats of shape
            (src_len = sentence_length, b = batch_size, e_word = embed_size),
            containing the CNN-based embeddings for each word of the sentences
            in the batch.
        """
        # A4 code
        # output = self.embeddings(input)
        # return output
        # End A4 code

        # YOUR CODE HERE for part 1j

        # 1. Record dimensions.
        src_len, b, _ = input.shape

        # 2. For each character lookup a dense character embedding. This gives
        #    a four-dimensional tensor of embeddings of shape:
        emb = self.char_embed(input)  # Tensor: (src_len, b, m_word, e_char)
        # Tensor: (src_len * b, m_word, e_char)
        emb_reshaped = emb \
            .reshape(emb.size(0) * emb.size(1), emb.size(2), emb.size(3)) \
            .permute(0, 2, 1)         # Tensor: (src_len * b, e_char, m_word)

        # 3. Combine character embeddings using 1-dimensional convolutions.
        conv_out = self.cnn(emb_reshaped)  # Tensor: (src_len * b, e_word)

        # 4. Run conv_out through highway network to get final word embeddings
        #    and top off with a dropout.
        highway = self.highway(conv_out)  # Tensor: (src_len * b, e_word)
        word_emb = self.dropout(highway)  # Tensor: (src_len * b, e_word)

        # 5. Reshape word embeddings to expected output shape.
        output = word_emb \
            .reshape(src_len, b, word_emb.size(1))  # Tensor: (src_len, b, e_word)

        # END YOUR CODE

        return output
