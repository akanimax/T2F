""" Module implementing the Conditional GAN which will be trained using the Progressive growing
    technique -> https://arxiv.org/abs/1710.10196
"""

import torch as th
import numpy as np


class Generator(th.nn.Module):
    """ Generator of the GAN network """

    class InitialBlock(th.nn.Module):
        """ Module implementing the initial block of the input """

        def __init__(self, in_channels):
            from torch.nn import ConvTranspose2d, Conv2d, LeakyReLU
            from torch.nn.functional import local_response_norm

            super().__init__()

            self.conv_1 = ConvTranspose2d(in_channels, in_channels, (4, 4), bias=False)
            self.conv_2 = Conv2d(in_channels, in_channels, (3, 3), padding=1, bias=False)

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
            self.conv_1 = Conv2d(in_channels, out_channels, (3, 3), padding=1, bias=False)
            self.conv_2 = Conv2d(out_channels, out_channels, (3, 3), padding=1, bias=False)

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


if __name__ == '__main__':

    from data_processing.DataLoader import Face2TextDataset, get_data_loader, get_transform
    from networks.TextEncoder import Encoder
    from networks.ConditionAugmentation import ConditionAugmentor

    import cv2
    import matplotlib.pyplot as plt

    dataset = Face2TextDataset(
        "../processed_annotations/processed_text.pkl",
        "../../data/LFW/lfw",
        img_transform=get_transform((256, 256))
    )

    dl = get_data_loader(dataset, 16, 3)

    batch = iter(dl).next()

    captions, images = batch
    random_caption = captions[1: 3]

    # print(random_caption)

    # perform the forward pass of the network
    encoder = Encoder(128, len(dataset.vocab), 256, 3)
    gan_input = encoder(random_caption)
    # print(gan_input.shape)
    # print(gan_input)

    # perform conditioning augmentation
    c_augmentor = ConditionAugmentor(gan_input.shape[-1], latent_size=256)
    c_not_hats = c_augmentor(gan_input)

    print(c_not_hats)
    print("C_not_hats:", c_not_hats.shape)

    # create random noise:
    z = th.randn(2, 256)
    gan_input = th.cat((c_not_hats, z), dim=-1)
    print("GAN input shape:", gan_input.shape)

    # perform forward pass on it:
    generator = Generator(latent_size=64)
    print("performing gan forward pass ...")
    out = generator(gan_input[:, :64], 6, 0.3)

    print("displaying image by resizing ...")
    img = (out[0].permute(1, 2, 0).detach().numpy() / 2) + 0.5
    print(img.shape)
    plt.imshow(cv2.resize(img, (256, 256)))
    plt.show()
