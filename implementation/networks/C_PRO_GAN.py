""" Module implementing the Conditional GAN which will be trained using the Progressive growing
    technique -> https://arxiv.org/abs/1710.10196
"""

import numpy as np
import torch as th


class Generator(th.nn.Module):
    """ Generator of the GAN network """

    class InitialBlock(th.nn.Module):
        """ Module implementing the initial block of the input """

        def __init__(self, in_channels):
            from torch.nn import ConvTranspose2d, Conv2d, LeakyReLU
            from torch.nn.functional import local_response_norm

            super().__init__()

            self.conv_1 = ConvTranspose2d(in_channels, in_channels, (4, 4))
            self.conv_2 = Conv2d(in_channels, in_channels, (3, 3), padding=1)

            # Pixelwise feature vector normalization operation
            self.pixNorm = lambda x: local_response_norm(x, 2 * x.shape[1], alpha=2, beta=0.5,
                                                         k=1e-8)

            # leaky_relu:
            self.lrelu = LeakyReLU(0.2)

        def forward(self, x):
            # convert the tensor shape:
            y = th.unsqueeze(th.unsqueeze(x, -1), -1)

            # perform the forward computations:
            y = self.lrelu(self.conv_1(y))
            y = self.lrelu(self.conv_2(y))

            # apply pixel norm
            y = self.pixNorm(y)

            return y

    class GeneralConvBlock(th.nn.Module):

        def __init__(self, in_channels, out_channels):
            from torch.nn import Conv2d, LeakyReLU, Upsample
            from torch.nn.functional import local_response_norm

            super().__init__()

            self.upsample = Upsample(scale_factor=2)
            self.conv_1 = Conv2d(in_channels, out_channels, (3, 3), padding=1)
            self.conv_2 = Conv2d(out_channels, out_channels, (3, 3), padding=1)

            # Pixelwise feature vector normalization operation
            self.pixNorm = lambda x: local_response_norm(x, 2 * x.shape[1], alpha=2, beta=0.5,
                                                         k=1e-8)

            # leaky_relu:
            self.lrelu = LeakyReLU(0.2)

        def forward(self, x):
            y = self.upsample(x)
            y = self.pixNorm(self.lrelu(self.conv_1(y)))
            y = self.pixNorm(self.lrelu(self.conv_2(y)))

            return y

    def __init__(self, depth=7, latent_size=512):

        from torch.nn import Conv2d, ModuleList, Upsample

        super(Generator, self).__init__()

        assert latent_size != 0 and ((latent_size & (latent_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert latent_size >= np.power(2, depth - 4), "latent size will diminish to zero"

        # state of the generator:
        self.depth = depth
        self.latent_size = latent_size

        # register the modules required for the GAN
        self.initial_block = self.InitialBlock(self.latent_size)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([])  # initialize to empty list

        # create the ToRGB layers for various outputs:
        self.toRGB = lambda in_channels: Conv2d(in_channels, 3, (1, 1), bias=False)
        self.rgb_converters = ModuleList([self.toRGB(self.latent_size)])

        # create the remaining layers
        for i in range(self.depth - 1):
            if i <= 2:
                layer = self.GeneralConvBlock(self.latent_size, self.latent_size)
                rgb = self.toRGB(self.latent_size)
            else:
                layer = self.GeneralConvBlock(
                    int(self.latent_size // np.power(2, i - 3)),
                    int(self.latent_size // np.power(2, i - 2))
                )
                rgb = self.toRGB(int(self.latent_size // np.power(2, i - 2)))
            self.layers.append(layer)
            self.rgb_converters.append(rgb)

        # register the temporary upsampler
        self.temporaryUpsampler = Upsample(scale_factor=2)

    def forward(self, x, depth, alpha):
        from torch.nn.functional import tanh

        assert depth < self.depth, "Requested output depth cannot be produced"

        y = self.initial_block(x)

        if depth > 0:
            for block in self.layers[:depth - 1]:
                y = block(y)

            residual = tanh(self.rgb_converters[depth - 1](self.temporaryUpsampler(y)))
            straight = tanh(self.rgb_converters[depth](self.layers[depth - 1](y)))

            out = (alpha * straight) + ((1 - alpha) * residual)

        else:
            out = tanh(self.rgb_converters[0](y))

        return out


class Discriminator(th.nn.Module):
    """ Discriminator of the GAN """

    class FinalBlock(th.nn.Module):
        """ Initial block for the Discriminator """

        class MinibatchStdDev(th.nn.Module):
            """ module implementing the minibatch_Stddev from the Pro-GAN paper. """

            def __init__(self):
                """ constructor for the class """
                super().__init__()
                # this layer doesn't have parameters

            def forward(self, x):
                """
                forward pass of the module
                :param x: input Tensor (B x C x H x W)
                :return: fwd => output Tensor (B x (C + 1) x H x W)
                """

                # calculate the std of x over the batch dimension
                std_x = x.std(dim=0)

                # average the std over all
                m_value = std_x.mean()

                # replicate the value over all spatial locations for
                # all examples
                b_size, _, h, w = x.shape
                constant_concat = m_value.expand(b_size, 1, h, w)
                fwd = th.cat((x, constant_concat), dim=1)

                # return the output tensor
                return fwd

        def __init__(self, in_channels, embedding_size):
            """
            constructor of the class
            :param in_channels: number of input channels
            :param embedding_size: size of the embedding
            """
            from torch.nn import Conv2d, LeakyReLU

            super().__init__()

            # declare the required modules for forward pass
            self.batch_discriminator = self.MinibatchStdDev()
            self.conv_1 = Conv2d(in_channels + 1, in_channels, (3, 3), padding=1)
            self.conv_2 = Conv2d(in_channels, in_channels, (4, 4))

            # final conv layer emulates a fully connected layer
            self.conv_3 = Conv2d(in_channels + embedding_size, 1, (1, 1))

            # leaky_relu:
            self.lrelu = LeakyReLU(0.2)

        def forward(self, x, embedding):
            """
            forward pass of the FinalBlock
            :param x: input
            :param embedding: embedding (for conditional GAN)
            :return: y => output
            """
            # minibatch_std_dev layer
            y = self.batch_discriminator(x)

            # define the computations
            y = self.lrelu(self.conv_1(y))
            y = self.lrelu(self.conv_2(y))

            # concatenate the text embedding to the features before the last
            # fully connected layer
            y = th.cat((y, th.unsqueeze(th.unsqueeze(embedding, -1), -1)), dim=1)
            y = self.lrelu(self.conv_3(y))  # final fully connected layer

            # flatten the output raw discriminator scores
            return y.view(-1)

    class GeneralConvBlock(th.nn.Module):
        """ General block in the discriminator  """

        def __init__(self, in_channels, out_channels):
            """
            constructor of the class
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            """
            from torch.nn import Conv2d, AvgPool2d, LeakyReLU

            super().__init__()

            self.conv_1 = Conv2d(in_channels, in_channels, (3, 3), padding=1)
            self.conv_2 = Conv2d(in_channels, out_channels, (3, 3), padding=1)
            self.downSampler = AvgPool2d(2)

            # leaky_relu:
            self.lrelu = LeakyReLU(0.2)

        def forward(self, x):
            """
            forward pass of the module
            :param x: input
            :return: y => output
            """
            # define the computations
            y = self.lrelu(self.conv_1(x))
            y = self.lrelu(self.conv_2(y))
            y = self.downSampler(y)

            return y

    def __init__(self, embedding_size, height=7, feature_size=512):
        """
        constructor for the class
        :param embedding_size: embedding_size for conditional discrimination
        :param height: total height of the discriminator (Must be equal to the Generator depth)
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        """
        from torch.nn import ModuleList, Conv2d, AvgPool2d

        super(Discriminator, self).__init__()

        assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if height >= 4:
            assert feature_size >= np.power(2, height - 4), "feature size cannot be produced"

        # create state of the object
        self.height = height
        self.feature_size = feature_size
        self.embedding_size = embedding_size

        self.final_block = self.FinalBlock(self.feature_size, embedding_size)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([])  # initialize to empty list

        # create the fromRGB layers for various inputs:
        self.fromRGB = lambda out_channels: Conv2d(3, out_channels, (1, 1), bias=False)
        self.rgb_to_features = ModuleList([self.fromRGB(self.feature_size)])

        # create the remaining layers
        for i in range(self.height - 1):
            if i > 2:
                layer = self.GeneralConvBlock(
                    int(self.feature_size // np.power(2, i - 2)),
                    int(self.feature_size // np.power(2, i - 3))
                )
                rgb = self.fromRGB(int(self.feature_size // np.power(2, i - 2)))
            else:
                layer = self.GeneralConvBlock(self.feature_size, self.feature_size)
                rgb = self.fromRGB(self.feature_size)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # register the temporary downSampler
        self.temporaryDownsampler = AvgPool2d(2)

    def forward(self, x, embedding, height, alpha):
        """
        forward pass of the discriminator
        :param x: input to the network
        :param embedding: conditional discrimination
        :param height: current height of operation (Progressive GAN)
        :param alpha: current value of alpha for fade-in
        :return: out => raw prediction values (WGAN-GP)
        """
        assert height < self.height, "Requested output depth cannot be produced"

        if height > 0:
            residual = self.rgb_to_features[height - 1](self.temporaryDownsampler(x))
            straight = self.layers[height - 1](
                self.rgb_to_features[height](x)
            )

            y = (alpha * straight) + ((1 - alpha) * residual)

            for block in reversed(self.layers[:height - 1]):
                y = block(y)
        else:
            y = self.rgb_to_features[0](x)

        out = self.final_block(y, embedding)

        return out


class ProGAN:
    """ Wrapper around the Generator and the Discriminator """

    def __init__(self, embedding_size, depth=7, latent_size=64, learning_rate=0.001, beta_1=0,
                 beta_2=0.99, eps=1e-8, drift=0.001, n_critic=1, device=th.device("cpu")):
        """
        constructor for the class
        :param embedding_size: embedding size for the conditional output
        :param depth: depth of the GAN (will be used for each generator and discriminator)
        :param latent_size: latent size of the manifold used by the GAN
        :param learning_rate: learning rate for Adam
        :param beta_1: beta_1 for Adam
        :param beta_2: beta_2 for Adam
        :param eps: epsilon for Adam
        :param n_critic: number of times to update discriminator
        :param device: device to run the GAN on (GPU / CPU)
        """

        from torch.optim import Adam

        # Create the Generator and the Discriminator
        self.gen = Generator(depth, latent_size).to(device)
        self.dis = Discriminator(embedding_size, depth, latent_size).to(device)

        # state of the object
        self.latent_size = latent_size
        self.embedding_size = embedding_size
        self.depth = depth
        self.n_critic = n_critic
        self.device = device
        self.drift = drift

        # define the optimizers for the discriminator and generator
        self.gen_optim = Adam(self.gen.parameters(), lr=learning_rate,
                              betas=(beta_1, beta_2), eps=eps)

        self.dis_optim = Adam(self.dis.parameters(), lr=learning_rate,
                              betas=(beta_1, beta_2), eps=eps)

    def __gradient_penalty(self, real_samps, fake_samps, embedding,
                           depth, alpha, reg_lambda=10):
        """
        private helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param embedding: text embedding for conditional discrimination
        :param depth: current depth in the optimization
        :param alpha: current alpha for fade-in
        :param reg_lambda: regularisation lambda
        :return: tensor (gradient penalty)
        """
        from torch.autograd import grad

        batch_size = real_samps.shape[0]

        # generate random epsilon
        epsilon = th.rand((batch_size, 1, 1, 1)).to(self.device)

        # create the merge of both real and fake samples
        merged = (epsilon * real_samps) + ((1 - epsilon) * fake_samps)

        # forward pass
        op = self.dis.forward(merged, embedding, depth, alpha)

        # obtain gradient of op wrt. merged
        gradient = grad(outputs=op, inputs=merged, create_graph=True,
                        grad_outputs=th.ones_like(op),
                        retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def __turn_off_dis_grads(self):
        """
        turn off discriminator gradients (to save computational power)
        :return: None
        """
        for p in self.dis.parameters():
            p.requires_grad = False

    def __turn_on_dis_grads(self):
        """
        turn on discriminator gradients (for weight updates)
        :return: None
        """
        for p in self.dis.parameters():
            p.requires_grad = True

    def optimize_discriminator(self, noise, embeddings, real_batch, depth, alpha,
                               use_matching_aware=True):
        """
        performs one step of weight update on discriminator using the batch of data
        :param noise: input noise of sample generation
        :param embeddings: text embeddings for conditional discrimination
        :param real_batch: real samples batch
        :param depth: current depth of optimization
        :param alpha: current alpha for fade-in
        :param use_matching_aware: whether to use a matching aware discriminator or not
        :return: current loss (Wasserstein loss)
        """
        from torch.nn import AvgPool2d

        # turn on gradients for discriminator
        self.__turn_on_dis_grads()

        # downsample the real_batch for the given depth
        down_sample_factor = int(np.power(2, self.depth - depth - 1))
        real_samples = AvgPool2d(down_sample_factor)(real_batch)

        loss_val = 0
        for _ in range(self.n_critic):
            # generate a batch of samples
            fake_samples = self.gen(noise, depth, alpha)

            # calculate the WGAN-GP (gradient penalty)
            gp = self.__gradient_penalty(real_samples, fake_samples, embeddings,
                                         depth, alpha)

            # calculate the matching aware distribution loss
            mis_match_text = embeddings[np.random.permutation(embeddings.shape[0]), :]
            m_a_d = self.dis(real_samples, mis_match_text, depth, alpha) if use_matching_aware \
                else 0

            # define the (Wasserstein) loss
            fake_out = self.dis(fake_samples, embeddings, depth, alpha)
            real_out = self.dis(real_samples, embeddings, depth, alpha)
            loss = (th.mean(fake_out) + th.mean(m_a_d) - th.mean(real_out)
                    + gp + (self.drift * th.mean(real_out ** 2)))

            # optimize discriminator
            self.dis_optim.zero_grad()
            loss.backward(retain_graph=True)
            self.dis_optim.step()

            loss_val += loss.item()

        return loss_val / self.n_critic

    def optimize_generator(self, noise, embeddings, depth, alpha):
        """
        performs one step of weight update on generator for the given batch_size
        :param noise: input random / conditional noise required for generating samples
        :param embeddings: text embeddings for conditional discrimination
        :param depth: depth of the network at which optimization is done
        :param alpha: value of alpha for fade-in effect
        :return: current loss (Wasserstein estimate)
        """
        # turn off discriminator gradient computations
        self.__turn_off_dis_grads()

        # generate fake samples:
        fake_samples = self.gen(noise, depth, alpha)

        loss = -th.mean(self.dis(fake_samples, embeddings, depth, alpha))

        # optimize the generator
        self.gen_optim.zero_grad()
        loss.backward(retain_graph=True)
        self.gen_optim.step()

        # return the loss value
        return loss.item()
