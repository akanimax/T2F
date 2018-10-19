""" Script for end-to-end training of the T2F model """
import datetime
import time
import torch as th
import numpy as np
import data_processing.DataLoader as dl
import argparse
import yaml
import os
import pickle
import timeit

from torch.backends import cudnn

# define the device for the training script
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# set torch manual seed for consistent output
th.manual_seed(3)

# Start fast training mode:
cudnn.benchmark = True


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="store", type=str, default="configs/1.conf",
                        help="default configuration for the Network")
    parser.add_argument("--start_depth", action="store", type=int, default=0,
                        help="Starting depth for training the network")
    parser.add_argument("--encoder_file", action="store", type=str, default=None,
                        help="pretrained Encoder file (compatible with my code)")
    parser.add_argument("--ca_file", action="store", type=str, default=None,
                        help="pretrained Conditioning Augmentor file (compatible with my code)")
    parser.add_argument("--generator_file", action="store", type=str, default=None,
                        help="pretrained Generator file (compatible with my code)")
    parser.add_argument("--discriminator_file", action="store", type=str, default=None,
                        help="pretrained Discriminator file (compatible with my code)")

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


def create_grid(samples, scale_factor, img_file, real_imgs=False):
    """
    utility function to create a grid of GAN samples
    :param samples: generated samples for storing
    :param scale_factor: factor for upscaling the image
    :param img_file: name of file to write
    :param real_imgs: turn off the scaling of images
    :return: None (saves a file)
    """
    from torchvision.utils import save_image
    from torch.nn.functional import interpolate

    samples = th.clamp((samples / 2) + 0.5, min=0, max=1)

    # upsample the image
    if not real_imgs and scale_factor > 1:
        samples = interpolate(samples,
                              scale_factor=scale_factor)

    # save the images:
    save_image(samples, img_file, nrow=int(np.sqrt(len(samples))))


def create_descriptions_file(file, captions, dataset):
    """
    utility function to create a file for storing the captions
    :param file: file for storing the captions
    :param captions: encoded_captions or raw captions
    :param dataset: the dataset object for transforming captions
    :return: None (saves a file)
    """
    from functools import reduce

    # transform the captions to text:
    if isinstance(captions, th.Tensor):
        captions = list(map(lambda x: dataset.get_english_caption(x.cpu()),
                            [captions[i] for i in range(captions.shape[0])]))

        with open(file, "w") as filler:
            for caption in captions:
                filler.write(reduce(lambda x, y: x + " " + y, caption, ""))
                filler.write("\n\n")
    else:
        with open(file, "w") as filler:
            for caption in captions:
                filler.write(caption)
                filler.write("\n\n")


def train_networks(encoder, ca, c_pro_gan, dataset, epochs,
                   encoder_optim, ca_optim, fade_in_percentage,
                   batch_sizes, start_depth, num_workers, feedback_factor,
                   log_dir, sample_dir, checkpoint_factor,
                   save_dir, use_matching_aware_dis=True):
    # required only for type checking
    from networks.TextEncoder import PretrainedEncoder

    # input assertions
    assert c_pro_gan.depth == len(batch_sizes), "batch_sizes not compatible with depth"
    assert c_pro_gan.depth == len(epochs), "epochs_sizes not compatible with depth"
    assert c_pro_gan.depth == len(fade_in_percentage), "fip_sizes not compatible with depth"

    # put all the Networks in training mode:
    ca.train()
    c_pro_gan.gen.train()
    c_pro_gan.dis.train()

    if not isinstance(encoder, PretrainedEncoder):
        encoder.train()

    print("Starting the training process ... ")

    # create fixed_input for debugging
    temp_data = dl.get_data_loader(dataset, batch_sizes[start_depth], num_workers=3)
    fixed_captions, fixed_real_images = iter(temp_data).next()
    fixed_embeddings = encoder(fixed_captions)
    fixed_embeddings = th.from_numpy(fixed_embeddings).to(device)

    fixed_c_not_hats, _, _ = ca(fixed_embeddings)

    fixed_noise = th.randn(len(fixed_captions),
                           c_pro_gan.latent_size - fixed_c_not_hats.shape[-1]).to(device)

    fixed_gan_input = th.cat((fixed_c_not_hats, fixed_noise), dim=-1)

    # save the fixed_images once:
    fixed_save_dir = os.path.join(sample_dir, "__Real_Info")
    os.makedirs(fixed_save_dir, exist_ok=True)
    create_grid(fixed_real_images, None,  # scale factor is not required here
                os.path.join(fixed_save_dir, "real_samples.png"), real_imgs=True)
    create_descriptions_file(os.path.join(fixed_save_dir, "real_captions.txt"),
                             fixed_captions,
                             dataset)

    # create a global time counter
    global_time = time.time()

    # delete temp data loader:
    del temp_data

    for current_depth in range(start_depth, c_pro_gan.depth):

        print("\n\nCurrently working on Depth: ", current_depth)
        current_res = np.power(2, current_depth + 2)
        print("Current resolution: %d x %d" % (current_res, current_res))

        data = dl.get_data_loader(dataset, batch_sizes[current_depth], num_workers)

        ticker = 1

        for epoch in range(1, epochs[current_depth] + 1):
            start = timeit.default_timer()  # record time at the start of epoch

            print("\nEpoch: %d" % epoch)
            total_batches = len(iter(data))
            fader_point = int((fade_in_percentage[current_depth] / 100)
                              * epochs[current_depth] * total_batches)

            for (i, batch) in enumerate(data, 1):
                # calculate the alpha for fading in the layers
                alpha = ticker / fader_point if ticker <= fader_point else 1

                # extract current batch of data for training
                captions, images = batch

                if encoder_optim is not None:
                    captions = captions.to(device)

                images = images.to(device)

                # perform text_work:
                embeddings = th.from_numpy(encoder(captions)).to(device)
                if encoder_optim is None:
                    # detach the LSTM from backpropagation
                    embeddings = embeddings.detach()
                c_not_hats, mus, sigmas = ca(embeddings)

                z = th.randn(
                    len(captions),
                    c_pro_gan.latent_size - c_not_hats.shape[-1]
                ).to(device)

                gan_input = th.cat((c_not_hats, z), dim=-1)

                # optimize the discriminator:
                dis_loss = c_pro_gan.optimize_discriminator(gan_input, images,
                                                            embeddings.detach(),
                                                            current_depth, alpha,
                                                            use_matching_aware_dis)

                # optimize the generator:
                z = th.randn(
                    captions.shape[0] if isinstance(captions, th.Tensor) else len(captions),
                    c_pro_gan.latent_size - c_not_hats.shape[-1]
                ).to(device)

                gan_input = th.cat((c_not_hats, z), dim=-1)

                if encoder_optim is not None:
                    encoder_optim.zero_grad()

                ca_optim.zero_grad()
                gen_loss = c_pro_gan.optimize_generator(gan_input, embeddings,
                                                        current_depth, alpha)

                # once the optimize_generator is called, it also sends gradients
                # to the Conditioning Augmenter and the TextEncoder. Hence the
                # zero_grad statements prior to the optimize_generator call
                # now perform optimization on those two as well
                # obtain the loss (KL divergence from ca_optim)
                kl_loss = th.mean(0.5 * th.sum((mus ** 2) + (sigmas ** 2)
                                               - th.log((sigmas ** 2)) - 1, dim=1))
                kl_loss.backward()
                ca_optim.step()
                if encoder_optim is not None:
                    encoder_optim.step()

                # provide a loss feedback
                if i % int(total_batches / feedback_factor) == 0 or i == 1:
                    elapsed = time.time() - global_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print("Elapsed [%s]  batch: %d  d_loss: %f  g_loss: %f  kl_los: %f"
                          % (elapsed, i, dis_loss, gen_loss, kl_loss.item()))

                    # also write the losses to the log file:
                    os.makedirs(log_dir, exist_ok=True)
                    log_file = os.path.join(log_dir, "loss_" + str(current_depth) + ".log")
                    with open(log_file, "a") as log:
                        log.write(str(dis_loss) + "\t" + str(gen_loss)
                                  + "\t" + str(kl_loss.item()) + "\n")

                    # create a grid of samples and save it
                    gen_img_file = os.path.join(sample_dir, "gen_" + str(current_depth) +
                                                "_" + str(epoch) + "_" +
                                                str(i) + ".png")

                    create_grid(
                        samples=c_pro_gan.gen(
                            fixed_gan_input,
                            current_depth,
                            alpha
                        ),
                        scale_factor=int(np.power(2, c_pro_gan.depth - current_depth - 1)),
                        img_file=gen_img_file,
                    )

                # increment the ticker:
                ticker += 1

            stop = timeit.default_timer()
            print("Time taken for epoch: %.3f secs" % (stop - start))

            if epoch % checkpoint_factor == 0 or epoch == 0:
                # save the Model
                encoder_save_file = os.path.join(save_dir, "Encoder_" +
                                                 str(current_depth) + ".pth")
                ca_save_file = os.path.join(save_dir, "Condition_Augmentor_" +
                                            str(current_depth) + ".pth")
                gen_save_file = os.path.join(save_dir, "GAN_GEN_" +
                                             str(current_depth) + ".pth")
                dis_save_file = os.path.join(save_dir, "GAN_DIS_" +
                                             str(current_depth) + ".pth")

                os.makedirs(save_dir, exist_ok=True)

                if encoder_optim is not None:
                    th.save(encoder.state_dict(), encoder_save_file, pickle)
                th.save(ca.state_dict(), ca_save_file, pickle)
                th.save(c_pro_gan.gen.state_dict(), gen_save_file, pickle)
                th.save(c_pro_gan.dis.state_dict(), dis_save_file, pickle)

    print("Training completed ...")


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """

    from networks.TextEncoder import Encoder
    from networks.ConditionAugmentation import ConditionAugmentor
    from pro_gan_pytorch.PRO_GAN import ConditionalProGAN

    print(args.config)
    config = get_config(args.config)
    print("Current Configuration:", config)

    # create the dataset for training
    if config.use_pretrained_encoder:
        dataset = dl.RawTextFace2TextDataset(
            annots_file=config.annotations_file,
            img_dir=config.images_dir,
            img_transform=dl.get_transform(config.img_dims)
        )
        from networks.TextEncoder import PretrainedEncoder
        # create a new session object for the pretrained encoder:
        text_encoder = PretrainedEncoder(
            model_file=config.pretrained_encoder_file,
            embedding_file=config.pretrained_embedding_file,
            device=device
        )
        encoder_optim = None
    else:
        dataset = dl.Face2TextDataset(
            pro_pick_file=config.processed_text_file,
            img_dir=config.images_dir,
            img_transform=dl.get_transform(config.img_dims),
            captions_len=config.captions_length
        )
        text_encoder = Encoder(
            embedding_size=config.embedding_size,
            vocab_size=dataset.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            device=device
        )
        encoder_optim = th.optim.Adam(text_encoder.parameters(),
                                      lr=config.learning_rate,
                                      betas=(config.beta_1, config.beta_2),
                                      eps=config.eps)

    # create the networks

    if args.encoder_file is not None:
        # Note this should not be used with the pretrained encoder file
        print("Loading encoder from:", args.encoder_file)
        text_encoder.load_state_dict(th.load(args.encoder_file))

    condition_augmenter = ConditionAugmentor(
        input_size=config.hidden_size,
        latent_size=config.ca_out_size,
        use_eql=config.use_eql,
        device=device
    )

    if args.ca_file is not None:
        print("Loading conditioning augmenter from:", args.ca_file)
        condition_augmenter.load_state_dict(th.load(args.ca_file))

    c_pro_gan = ConditionalProGAN(
        embedding_size=config.hidden_size,
        depth=config.depth,
        latent_size=config.latent_size,
        compressed_latent_size=config.compressed_latent_size,
        learning_rate=config.learning_rate,
        beta_1=config.beta_1,
        beta_2=config.beta_2,
        eps=config.eps,
        drift=config.drift,
        n_critic=config.n_critic,
        use_eql=config.use_eql,
        loss=config.loss_function,
        use_ema=config.use_ema,
        ema_decay=config.ema_decay,
        device=device
    )

    if args.generator_file is not None:
        print("Loading generator from:", args.generator_file)
        c_pro_gan.gen.load_state_dict(th.load(args.generator_file))

    if args.discriminator_file is not None:
        print("Loading discriminator from:", args.discriminator_file)
        c_pro_gan.dis.load_state_dict(th.load(args.discriminator_file))

    # create the optimizer for Condition Augmenter separately
    ca_optim = th.optim.Adam(condition_augmenter.parameters(),
                             lr=config.learning_rate,
                             betas=(config.beta_1, config.beta_2),
                             eps=config.eps)

    print("Generator Config:")
    print(c_pro_gan.gen)

    print("\nDiscriminator Config:")
    print(c_pro_gan.dis)

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
        start_depth=args.start_depth,
        batch_sizes=config.batch_sizes,
        num_workers=config.num_workers,
        feedback_factor=config.feedback_factor,
        log_dir=config.log_dir,
        sample_dir=config.sample_dir,
        checkpoint_factor=config.checkpoint_factor,
        save_dir=config.save_dir,
        use_matching_aware_dis=config.use_matching_aware_discriminator
    )


if __name__ == '__main__':
    # invoke the main function of the script
    main(parse_arguments())
