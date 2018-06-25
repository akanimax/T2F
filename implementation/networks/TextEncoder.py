""" Module defining the text encoder used for conditioning the generation of the GAN """

import torch as th


class Encoder(th.nn.Module):
    """ Encodes the given text input into a high dimensional embedding vector
        uses LSTM internally
    """

    def __init__(self, embedding_size, vocab_size, hidden_size, num_layers):
        """
        constructor of the class
        :param embedding_size: size of the input embeddings
        :param vocab_size: size of the vocabulary
        :param hidden_size: hidden size of the LSTM network
        :param num_layers: number of LSTM layers in the network
        """
        super(Encoder, self).__init__()

        # create the state:
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # create the LSTM layer:
        from torch.nn import Embedding, Sequential, LSTM
        self. network = Sequential(
            Embedding(self.vocab_size, self.embedding_size, padding_idx=0),
            LSTM(self.embedding_size, self.hidden_size,
                 self.num_layers, batch_first=True)
        )

    def forward(self, x):
        """
        performs forward pass on the given data:
        :param x: input numeric sequence
        :return: enc_emb: encoded text embedding
        """
        _, (enc_emb, _) = self.network(x)
        return enc_emb[-1]  # return the deepest embedding
