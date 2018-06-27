""" Script for end-to-end training of the T2F model """

import torch as th
import numpy as np
import data_processing.DataLoader as dl
import argparse
import yaml
import os
import pickle

# define the device for the training script
device = th.device("cuda" if th.cuda.is_available() else "cpu")


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="store", type=str, default="configs/1.conf",
                        help="default configuration for the Network")
    args = parser.parse_args()

    return args


def get_config(conf_file):
    """
    parse and load the provided configuration
    :param conf_file: configuration file
    :return: conf => parsed configuration
    """
    from easydict import EasyDict as edict

    with open(conf_file, "r") as file_descriptor:
        data = yaml.load(file_descriptor)

    # convert the data into an easyDictionary
    return edict(data)


def create_grid(samples, scale_factor, img_file, width=2):
    """
    utility funtion to create a grid of GAN samples
    :param samples: generated samples for storing
    :param scale_factor: factor for upscaling the image
    :param img_file: name of file to write
    :param width: width for the grid
    :return: None (saves a file)
    """
    from torchvision.utils import save_image
    from torch.nn.functional import upsample

    # upsmaple the image
    if scale_factor > 1:
        samples = upsample(samples, scale_factor=scale_factor)

    # save the images:
    save_image(samples, img_file, nrow=width)


def train_networks(encoder, ca, c_pro_gan, dataset, epochs,
                   encoder_optim, ca_optim, fade_in_percentage,
                   batch_sizes, num_workers, feedback_factor,
                   log_file, sample_dir, checkpoint_factor,
                   save_dir):
    assert c_pro_gan.depth == len(batch_sizes), "batch_sizes not compatible with depth"

    print("Starting the training process ... ")
    for current_depth in range(c_pro_gan.depth):

        print("\n\nCurrently working on Depth: ", current_depth)
        current_res = np.power(2, current_depth + 2)
        print("Current resolution: %d x %d" % (current_res, current_res))

        data = dl.get_data_loader(dataset, batch_sizes[current_depth], num_workers)
        fader_point = int((fade_in_percentage / 100) * epochs)

        for epoch in range(1, epochs + 1):

            print("Epoch: %d" % epoch)
            total_batches = len(iter(data))

            for (i, batch) in enumerate(data, 1):
                # calculate the alpha for fading in the layers
                alpha = epoch / fader_point if epoch <= fader_point else 1

                # extract current batch of data for training
                captions, images = batch
                captions, images = captions.to(device), images.to(device)

                # perform text_work:
                embeddings = encoder(captions)
                c_not_hats = ca(embeddings)
                z = th.randn(
                    captions.shape[0],
                    c_pro_gan.latent_size - c_not_hats.shape[-1]
                ).to(device)

                gan_input = th.cat((c_not_hats, z), dim=-1)

                # optimize the discriminator:
                dis_loss = c_pro_gan.optimize_discriminator(gan_input, embeddings,
                                                            images, current_depth, alpha)

                # optimize the generator:
                encoder_optim.zero_grad()
                ca_optim.zero_grad()
                gen_loss = c_pro_gan.optimize_generator(gan_input, embeddings,
                                                        current_depth, alpha)

                # once the optimize_generator is called, it also sends gradients
                # to the Conditioning Augmenter and the TextEncoder. Hence the
                # zero_grad statements prior to the optimize_generator call
                # now perform optimization on those two as well
                ca_optim.step()
                encoder_optim.step()

                # provide a loss feedback
                if i % int(total_batches / feedback_factor) == 0 or i == 1:
                    print("batch: %d  d_loss: %f  g_loss: %f" % (i, dis_loss, gen_loss))

                    # also write the losses to the log file:
                    with open(log_file, "a") as log:
                        log.write(str(dis_loss) + "\t" + str(gen_loss) + "\n")

                    # create a grid of samples and save it
                    img_file = os.path.join(sample_dir, str(current_depth) + "_" + str(epoch) + "_" +
                                            str(i) + ".png")
                    create_grid(
                        samples=c_pro_gan.gen(
                            gan_input,
                            current_depth,
                            alpha
                        ),
                        scale_factor=int(np.power(2, c_pro_gan.depth - current_depth - 1)),
                        img_file=img_file,
                        width=int(np.sqrt(batch_sizes[current_depth])),
                    )

            if (epoch + 1) % checkpoint_factor == 0 or epoch == 0:
                # save the Model
                encoder_save_file = os.path.join(save_dir, "Encoder" +
                                                 str(current_depth) + "_" + str(epoch) + ".pth")
                ca_save_file = os.path.join(save_dir, "GAN_GEN" +
                                            str(current_depth) + "_" + str(epoch) + ".pth")
                gen_save_file = os.path.join(save_dir, "GAN_GEN" +
                                             str(current_depth) + "_" + str(epoch) + ".pth")
                dis_save_file = os.path.join(save_dir, "GAN_DIS" +
                                             str(current_depth) + "_" + str(epoch) + ".pth")

                th.save(encoder.state_dict(), encoder_save_file, pickle)
                th.save(ca.state_dict(), ca_save_file, pickle)
                th.save(c_pro_gan.gen.state_dict(), gen_save_file, pickle)
                th.save(c_pro_gan.dis.state_dict(), dis_save_file, pickle)

        print("Training completed ...")


def main(config):
    """
    Main function for the script
    :param config: configuration used for training the network
    :return: None
    """

    from networks.TextEncoder import Encoder
    from networks.ConditionAugmentation import ConditionAugmentor
    from networks.C_PRO_GAN import ProGAN

    print("Current Configuration:", config)

    # create the dataset for training
    dataset = dl.Face2TextDataset(
        pro_pick_file=config.processed_text_file,
        img_dir=config.images_dir,
        img_transform=dl.get_transform(config.img_dims),
        captions_len=config.captions_length
    )

    # create the networks
    text_encoder = Encoder(
        embedding_size=config.embedding_size,
        vocab_size=dataset.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        device=device
    )

    condition_augmenter = ConditionAugmentor(
        input_size=config.hidden_size,
        latent_size=config.ca_out_size,
        device=device
    )

    c_pro_gan = ProGAN(
        embedding_size=config.hidden_size,
        depth=config.depth,
        latent_size=config.latent_size,
        learning_rate=config.learning_rate,
        beta_1=config.beta_1,
        beta_2=config.beta_2,
        eps=config.eps,
        drift=config.drift,
        n_critic=config.n_critic,
        device=device
    )

    # create the optimizers for Encoder and Condition Augmenter separately
    encoder_optim = th.optim.Adam(text_encoder.parameters(),
                                  lr=config.learning_rate,
                                  betas=(config.beta_1, config.beta_2),
                                  eps=config.eps)

    ca_optim = th.optim.Adam(condition_augmenter.parameters(),
                             lr=config.learning_rate,
                             betas=(config.beta_1, config.beta_2),
                             eps=config.eps)

    # train all the networks
    train_networks(
        encoder=text_encoder,
        ca=condition_augmenter,
        c_pro_gan=c_pro_gan,
        dataset=dataset,
        encoder_optim=encoder_optim,
        ca_optim=ca_optim,
        epochs=config.epochs,
        fade_in_percentage=config.fade_in_percentage,
        batch_sizes=config.batch_sizes,
        num_workers=config.num_workers,
        feedback_factor=config.feedback_factor,
        log_file=config.log_file,
        sample_dir=config.sample_dir,
        checkpoint_factor=config.checkpoint_factor,
        save_dir=config.save_dir
    )


if __name__ == '__main__':
    # invoke the main function of the script
    main(get_config(parse_arguments().config))
