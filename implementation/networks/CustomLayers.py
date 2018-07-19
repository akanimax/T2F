""" Module containing custom layers """
import torch as th
import copy


# extending Conv2D and Deconv2D layers for equalized learning rate logic
class _equalized_conv2d(th.nn.Module):
    """ conv2d with the concept of equalized learning rate """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, initializer='kaiming', bias=True):
        """
        constructor for the class
        :param c_in: input channels
        :param c_out:  output channels
        :param k_size: kernel size (h, w) should be a tuple or a single integer
        :param stride: stride for conv
        :param pad: padding
        :param initializer: initializer. one of kaiming or xavier
        :param bias: whether to use bias or not
        """
        super(_equalized_conv2d, self).__init__()
        self.conv = th.nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=True)
        if initializer == 'kaiming':
            th.nn.init.kaiming_normal_(self.conv.weight, a=th.nn.init.calculate_gain('conv2d'))
        elif initializer == 'xavier':
            th.nn.init.xavier_normal_(self.conv.weight)

        self.use_bias = bias

        self.bias = th.nn.Parameter(th.FloatTensor(c_out).fill_(0))
        self.scale = (th.mean(self.conv.weight.data ** 2)) ** 0.5
        self.conv.weight.data.copy_(self.conv.weight.data / self.scale)

    def forward(self, x):
        """
        forward pass of the network
        :param x: input
        :return: y => output
        """
        try:
            dev_scale = self.scale.to(x.get_device())
        except RuntimeError:
            dev_scale = self.scale
        x = self.conv(x.mul(dev_scale))
        if self.use_bias:
            return x + self.bias.view(1, -1, 1, 1).expand_as(x)
        return x


class _equalized_deconv2d(th.nn.Module):
    """ Transpose convolution using the equalized learning rate """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, initializer='kaiming', bias=True):
        """
        constructor for the class
        :param c_in: input channels
        :param c_out: output channels
        :param k_size: kernel size
        :param stride: stride for convolution transpose
        :param pad: padding
        :param initializer: initializer. one of kaiming or xavier
        :param bias: whether to use bias or not
        """
        super(_equalized_deconv2d, self).__init__()
        self.deconv = th.nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False)
        if initializer == 'kaiming':
            th.nn.init.kaiming_normal_(self.deconv.weight, a=th.nn.init.calculate_gain('conv2d'))
        elif initializer == 'xavier':
            th.nn.init.xavier_normal_(self.deconv.weight)

        self.use_bias = bias

        self.bias = th.nn.Parameter(th.FloatTensor(c_out).fill_(0))
        self.scale = (th.mean(self.deconv.weight.data ** 2)) ** 0.5
        self.deconv.weight.data.copy_(self.deconv.weight.data / self.scale)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        try:
            dev_scale = self.scale.to(x.get_device())
        except RuntimeError:
            dev_scale = self.scale

        x = self.deconv(x.mul(dev_scale))
        if self.use_bias:
            return x + self.bias.view(1, -1, 1, 1).expand_as(x)
        return x


class _equalized_linear(th.nn.Module):
    """ Linear layer using equalized learning rate """

    def __init__(self, c_in, c_out, initializer='kaiming', bias=True):
        """
        Linear layer from pytorch extended to include equalized learning rate
        :param c_in: number of input channels
        :param c_out: number of output channels
        :param initializer: initializer to be used: one of "kaiming" or "xavier"
        :param bias: whether to use bias with the linear layer
        """
        super(_equalized_linear, self).__init__()
        self.linear = th.nn.Linear(c_in, c_out, bias=bias)
        if initializer == 'kaiming':
            th.nn.init.kaiming_normal_(self.linear.weight,
                                       a=th.nn.init.calculate_gain('linear'))
        elif initializer == 'xavier':
            th.nn.init.xavier_normal_(self.linear.weight)

        self.use_bias = bias
        self.bias = th.nn.Parameter(th.FloatTensor(c_out).fill_(0))
        self.scale = (th.mean(self.linear.weight.data ** 2)) ** 0.5
        self.linear.weight.data.copy_(self.linear.weight.data / self.scale)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        try:
            dev_scale = self.scale.to(x.get_device())
        except RuntimeError:
            dev_scale = self.scale
        x = self.linear(x.mul(dev_scale))
        if self.use_bias:
            return x + self.bias.view(1, -1).expand_as(x)
        return x


# ==========================================================
# Layers required for Building The generator and
# discriminator
# ==========================================================
class GenInitialBlock(th.nn.Module):
    """ Module implementing the initial block of the input """

    def __init__(self, in_channels, use_eql):
        """
        constructor for the inner class
        :param in_channels: number of input channels to the block
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import LeakyReLU
        from torch.nn.functional import local_response_norm

        super(GenInitialBlock, self).__init__()

        if use_eql:
            self.conv_1 = _equalized_deconv2d(in_channels, in_channels, (4, 4), bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, in_channels, (3, 3),
                                            pad=1, bias=True)

        else:
            from torch.nn import Conv2d, ConvTranspose2d
            self.conv_1 = ConvTranspose2d(in_channels, in_channels, (4, 4), bias=True)
            self.conv_2 = Conv2d(in_channels, in_channels, (3, 3), padding=1, bias=True)

        # Pixelwise feature vector normalization operation
        self.pixNorm = lambda x: local_response_norm(x, 2 * x.shape[1], alpha=2,
                                                     beta=0.5, k=1e-8)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input to the module
        :return: y => output
        """
        # convert the tensor shape:
        y = th.unsqueeze(th.unsqueeze(x, -1), -1)

        # perform the forward computations:
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # apply pixel norm
        y = self.pixNorm(y)

        return y


class GenGeneralConvBlock(th.nn.Module):
    """ Module implementing a general convolutional block """

    def __init__(self, in_channels, out_channels, use_eql):
        """
        constructor for the class
        :param in_channels: number of input channels to the block
        :param out_channels: number of output channels required
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import LeakyReLU, Upsample
        from torch.nn.functional import local_response_norm

        super(GenGeneralConvBlock, self).__init__()

        self.upsample = Upsample(scale_factor=2)

        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels, out_channels, (3, 3),
                                            pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(out_channels, out_channels, (3, 3),
                                            pad=1, bias=True)
        else:
            from torch.nn import Conv2d
            self.conv_1 = Conv2d(in_channels, out_channels, (3, 3),
                                 padding=1, bias=True)
            self.conv_2 = Conv2d(out_channels, out_channels, (3, 3),
                                 padding=1, bias=True)

        # Pixelwise feature vector normalization operation
        self.pixNorm = lambda x: local_response_norm(x, 2 * x.shape[1], alpha=2,
                                                     beta=0.5, k=1e-8)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """
        y = self.upsample(x)
        y = self.pixNorm(self.lrelu(self.conv_1(y)))
        y = self.pixNorm(self.lrelu(self.conv_2(y)))

        return y


class EMA(th.nn.Module):
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def forward(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


class MinibatchStdDev(th.nn.Module):
    def __init__(self, averaging='all'):
        """
        constructor for the class
        :param averaging: the averaging mode used for calculating the MinibatchStdDev
        """
        super(MinibatchStdDev, self).__init__()

        # lower case the passed parameter
        self.averaging = averaging.lower()

        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in \
                   ['all', 'flat', 'spatial', 'none', 'gpool'], \
                   'Invalid averaging mode %s' % self.averaging

        # calculate the std_dev in such a way that it doesn't result in 0
        # otherwise 0 norm operation's gradient is nan
        self.adjusted_std = lambda x, **kwargs: th.sqrt(
            th.mean((x - th.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)

    def forward(self, x):
        """
        forward pass of the Layer
        :param x: input
        :return: y => output
        """
        shape = list(x.size())
        target_shape = copy.deepcopy(shape)

        # compute the std's over the minibatch
        vals = self.adjusted_std(x, dim=0, keepdim=True)

        # perform averaging
        if self.averaging == 'all':
            target_shape[1] = 1
            vals = th.mean(vals, dim=1, keepdim=True)

        elif self.averaging == 'spatial':
            if len(shape) == 4:
                vals = th.mean(th.mean(vals, 2, keepdim=True), 3, keepdim=True)

        elif self.averaging == 'none':
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]

        elif self.averaging == 'gpool':
            if len(shape) == 4:
                vals = th.mean(th.mean(th.mean(x, 2, keepdim=True),
                                       3, keepdim=True), 0, keepdim=True)
        elif self.averaging == 'flat':
            target_shape[1] = 1
            vals = th.FloatTensor([self.adjusted_std(x)])

        else:  # self.averaging == 'group'
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1] /
                             self.n, self.shape[2], self.shape[3])
            vals = th.mean(vals, 0, keepdim=True).view(1, self.n, 1, 1)

        # spatial replication of the computed statistic
        vals = vals.expand(*target_shape)

        # concatenate the constant feature map to the input
        y = th.cat([x, vals], 1)

        # return the computed value
        return y


class DisFinalBlock(th.nn.Module):
    """ Final block for the Discriminator """

    def __init__(self, in_channels, use_eql):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import LeakyReLU

        super(DisFinalBlock, self).__init__()

        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()
        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels + 1, in_channels, (3, 3), pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, in_channels, (4, 4), bias=True)
            # final conv layer emulates a fully connected layer
            self.conv_3 = _equalized_conv2d(in_channels, 1, (1, 1), bias=True)
        else:
            from torch.nn import Conv2d
            self.conv_1 = Conv2d(in_channels + 1, in_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels, in_channels, (4, 4), bias=True)
            # final conv layer emulates a fully connected layer
            self.conv_3 = Conv2d(in_channels, 1, (1, 1), bias=True)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the FinalBlock
        :param x: input
        :return: y => output
        """
        # minibatch_std_dev layer
        y = self.batch_discriminator(x)

        # define the computations
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # fully connected layer
        y = self.conv_3(y)  # This layer has linear activation

        # flatten the output raw discriminator scores
        return y.view(-1)


class ConDisFinalBlock(th.nn.Module):
    """ Final block for the Conditional Discriminator """

    def __init__(self, in_channels, in_latent_size, out_latent_size, use_eql):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param in_latent_size: size of the input latent vectors
        :param out_latent_size: size of the transformed latent vectors
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import LeakyReLU

        super(ConDisFinalBlock, self).__init__()

        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()
        if use_eql:
            self.compressor = _equalized_linear(c_in=in_latent_size, c_out=out_latent_size)
            self.conv_1 = _equalized_conv2d(in_channels + 1, in_channels, (3, 3), pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels + out_latent_size,
                                            in_channels, (1, 1), bias=True)
            self.conv_3 = _equalized_conv2d(in_channels, in_channels, (4, 4), bias=True)
            # final conv layer emulates a fully connected layer
            self.conv_4 = _equalized_conv2d(in_channels, 1, (1, 1), bias=True)
        else:
            from torch.nn import Conv2d, Linear
            self.compressor = Linear(in_features=in_latent_size,
                                     out_features=out_latent_size, bias=True)
            self.conv_1 = Conv2d(in_channels + 1, in_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels + out_latent_size,
                                 in_channels, (1, 1), bias=True)
            self.conv_3 = Conv2d(in_channels, in_channels, (4, 4), bias=True)
            # final conv layer emulates a fully connected layer
            self.conv_4 = Conv2d(in_channels, 1, (1, 1), bias=True)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x, latent_vector):
        """
        forward pass of the FinalBlock
        :param x: input
        :param latent_vector: latent vector for conditional discrimination
        :return: y => output
        """
        # minibatch_std_dev layer
        y = self.batch_discriminator(x)

        # define the computations
        y = self.lrelu(self.conv_1(y))
        # apply the latent vector here:
        compressed_latent_vector = self.compressor(latent_vector)
        cat = th.unsqueeze(th.unsqueeze(compressed_latent_vector, -1), -1)
        cat = cat.expand(
            compressed_latent_vector.shape[0],
            compressed_latent_vector.shape[1],
            y.shape[2],
            y.shape[3]
        )
        y = th.cat((y, cat), dim=1)

        y = self.lrelu(self.conv_2(y))
        y = self.lrelu(self.conv_3(y))

        # fully connected layer
        y = self.conv_4(y)  # This layer has linear activation

        # flatten the output raw discriminator scores
        return y.view(-1)


class DisGeneralConvBlock(th.nn.Module):
    """ General block in the discriminator  """

    def __init__(self, in_channels, out_channels, use_eql):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import AvgPool2d, LeakyReLU

        super(DisGeneralConvBlock, self).__init__()

        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels, in_channels, (3, 3), pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, out_channels, (3, 3), pad=1, bias=True)
        else:
            from torch.nn import Conv2d
            self.conv_1 = Conv2d(in_channels, in_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels, out_channels, (3, 3), padding=1, bias=True)

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
