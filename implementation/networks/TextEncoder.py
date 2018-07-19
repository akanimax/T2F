""" Module defining the text encoder used for conditioning the generation of the GAN """

import os

import tensorflow as tf
import tensorflow_hub as hub
import torch as th


class Encoder(th.nn.Module):
    """ Encodes the given text input into a high dimensional embedding vector
        uses LSTM internally
    """

    def __init__(self, embedding_size, vocab_size, hidden_size, num_layers, device=th.device("cpu")):
        """
        constructor of the class
        :param embedding_size: size of the input embeddings
        :param vocab_size: size of the vocabulary
        :param hidden_size: hidden size of the LSTM network
        :param num_layers: number of LSTM layers in the network
        :param device: device on which to run the Module
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
        ).to(device)

    def forward(self, x):
        """
        performs forward pass on the given data:
        :param x: input numeric sequence
        :return: enc_emb: encoded text embedding
        """
        output, (_, _) = self.network(x)
        return output[:, -1, :]  # return the deepest last (hidden state) embedding


class PretrainedEncoder:
    """
    Uses the TensorFlow Hub's module here ->
    https://www.tensorflow.org/hub/modules/google/universal-sentence-encoder/2
    """

    def __init__(self, session, module_dir=None, download=True):
        """
        constructor for the class
        :param session: TensorFlow session object
        :param module_dir: directory of an already downloaded module / where to download
        :param download: Boolean for whether to download
        """
        if module_dir is None:
            module_dir = "~/.tensorflow_hub_modules/text_encoder"

        self.download_path = \
            "https://tfhub.dev/google/universal-sentence-encoder/2"
        self.module_dir = module_dir
        self.session = session

        if download:
            os.environ['TFHUB_CACHE_DIR'] = module_dir
            self.module = hub.Module(self.download_path)

        else:
            self.module = hub.Module(self.module_dir)

        self.__run_initializers()

    def __run_initializers(self):
        """
        private helper method for initializing the graph with it's variables
        :return: None
        """
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.tables_initializer())

    def __call__(self, text_list):
        """
        encode the given texts into a summary embedding
        :param text_list: list[strings] (Note, this needs to be a list of strings not tokens)
        :return: embeddings => np array of shape => [*(variable) x embedding_size ()]
        """
        return self.session.run(self.module(text_list))
