""" Module defining the text encoder used for conditioning the generation of the GAN """

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
        self.network = Sequential(
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


class PretrainedEncoder(th.nn.Module):
    """
    Uses the Facebook's InferSent PyTorch module here ->
    https://github.com/facebookresearch/InferSent

    I have modified the implementation slightly in order to suit my use.
    Note that I am Giving proper Credit to the original
    InferSent Code authors by keeping a copy their LICENSE here.

    Unlike some people who have copied my code without regarding my LICENSE

    @Args:
        :param model_file: path to the pretrained '.pkl' model file
        :param embedding_file: path to the pretrained glove embeddings file
        :param vocab_size: size of the built vocabulary
                           default: 300000
        :param device: device to run the network on
                       default: "CPU"
    """

    def __init__(self, model_file, embedding_file,
                 vocab_size=300000, device=th.device("cpu")):
        """
        constructor of the class
        """
        from networks.InferSent.models import InferSent

        super().__init__()

        # this is fixed
        self.encoder = InferSent({
            'bsize': 64, 'word_emb_dim': 300,
            'enc_lstm_dim': 2048, 'pool_type': 'max',
            'dpout_model': 0.0, 'version': 2}).to(device)

        # load the model and embeddings into the model:
        self.encoder.load_state_dict(th.load(model_file))

        # load the vocabulary file and build the vocabulary
        self.encoder.set_w2v_path(embedding_file)
        self.encoder.build_vocab_k_words(vocab_size)

    def forward(self, x):
        """
        forward pass of the encoder
        :param x: input sentences to be encoded
                  list[Strings]
        :return: encodings for the sentences
                 shape => [batch_size x 4096]
        """

        # we just need the encodings here
        return self.encoder.encode(x, tokenize=False)[0]
