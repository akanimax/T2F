""" Module implementing various loss functions """

import torch as th


class GANLoss:
    """ Base class for all losses """

    def __init__(self, device, dis):
        self.device = device
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        raise NotImplementedError("gen_loss method has not been implemented")


class ConditionalGANLoss:
    """ Base class for all losses """

    def __init__(self, device, dis):
        self.device = device
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, latent_vector, height, alpha):
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, latent_vector, height, alpha):
        raise NotImplementedError("gen_loss method has not been implemented")


class WGAN_GP(GANLoss):

    def __init__(self, device, dis, drift=0.001, use_gp=False):
        super().__init__(device, dis)
        self.drift = drift
        self.use_gp = use_gp

    def __gradient_penalty(self, real_samps, fake_samps,
                           height, alpha, reg_lambda=10):
        """
        private helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param height: current depth in the optimization
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
        op = self.dis.forward(merged, height, alpha)

        # obtain gradient of op wrt. merged
        gradient = grad(outputs=op, inputs=merged, create_graph=True,
                        grad_outputs=th.ones_like(op),
                        retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # define the (Wasserstein) loss
        fake_out = self.dis(fake_samps, height, alpha)
        real_out = self.dis(real_samps, height, alpha)

        loss = (th.mean(fake_out) - th.mean(real_out)
                + (self.drift * th.mean(real_out ** 2)))

        if self.use_gp:
            # calculate the WGAN-GP (gradient penalty)
            fake_samps.requires_grad = True  # turn on gradients for penalty calculation
            gp = self.__gradient_penalty(real_samps, fake_samps, height, alpha)
            loss += gp

        return loss

    def gen_loss(self, _, fake_samps, height, alpha):
        # calculate the WGAN loss for generator
        loss = -th.mean(self.dis(fake_samps, height, alpha))

        return loss


class LSGAN(GANLoss):

    def __init__(self, device, dis):
        super().__init__(device, dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        return 0.5 * (((th.mean(self.dis(real_samps, height, alpha)) - 1) ** 2)
                      + (th.mean(self.dis(fake_samps, height, alpha))) ** 2)

    def gen_loss(self, _, fake_samps, height, alpha):
        return 0.5 * ((th.mean(self.dis(fake_samps, height, alpha)) - 1) ** 2)


class LSGAN_SIGMOID(GANLoss):

    def __init__(self, device, dis):
        super().__init__(device, dis)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        from torch.nn.functional import sigmoid
        real_scores = th.mean(sigmoid(self.dis(real_samps, height, alpha)))
        fake_scores = th.mean(sigmoid(self.dis(fake_samps, height, alpha)))
        return 0.5 * (((real_scores - 1) ** 2) + (fake_scores ** 2))

    def gen_loss(self, _, fake_samps, height, alpha):
        from torch.nn.functional import sigmoid
        scores = th.mean(sigmoid(self.dis(fake_samps, height, alpha)))
        return 0.5 * ((scores - 1) ** 2)


# =============================================================
# Conditional versions of the Losses:
# =============================================================

class CondWGAN_GP(ConditionalGANLoss):

    def __init__(self, device, dis, drift=0.001, use_gp=False):
        super().__init__(device, dis)
        self.drift = drift
        self.use_gp = use_gp

    def __gradient_penalty(self, real_samps, fake_samps, latent_vector,
                           height, alpha, reg_lambda=10):
        """
        private helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param latent_vector: used for conditional loss calculation
        :param height: current depth in the optimization
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
        op = self.dis.forward(merged, latent_vector, height, alpha)

        # obtain gradient of op wrt. merged
        gradient = grad(outputs=op, inputs=merged, create_graph=True,
                        grad_outputs=th.ones_like(op),
                        retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(self, real_samps, fake_samps, latent_vector, height, alpha):
        # define the (Wasserstein) loss
        fake_out = self.dis(fake_samps, latent_vector, height, alpha)
        real_out = self.dis(real_samps, latent_vector, height, alpha)

        loss = (th.mean(fake_out) - th.mean(real_out)
                + (self.drift * th.mean(real_out ** 2)))

        if self.use_gp:
            # calculate the WGAN-GP (gradient penalty)
            fake_samps.requires_grad = True  # turn on gradients for penalty calculation
            gp = self.__gradient_penalty(real_samps, fake_samps,
                                         latent_vector, height, alpha)
            loss += gp

        return loss

    def gen_loss(self, _, fake_samps, latent_vector, height, alpha):
        # calculate the WGAN loss for generator
        loss = -th.mean(self.dis(fake_samps, latent_vector, height, alpha))

        return loss
