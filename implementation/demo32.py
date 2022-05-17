import torch as th
import numpy as np
import data_processing.DataLoader as dl
import yaml


current_depth = 4

from networks.TextEncoder import Encoder
from networks.ConditionAugmentation import ConditionAugmentor
from networks.C_PRO_GAN import ProGAN

# define the device for the training script
device = th.device("cuda" if th.cuda.is_available() else "cpu")

############################################################################
#load my generator.


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

config = get_config("configs\\11.conf")

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

c_pro_gan.gen.load_state_dict(th.load("training_runs\\11\\saved_models\\GAN_GEN_3_20.pth"))

###################################################################################
#load my embedding and conditional augmentor

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
text_encoder.load_state_dict(th.load("training_runs\\11\\saved_models\\Encoder_3_20.pth"))

condition_augmenter = ConditionAugmentor(
        input_size=config.hidden_size,
        latent_size=config.ca_out_size,
        device=device
    )
condition_augmenter.load_state_dict(th.load("training_runs\\11\\saved_models\\Condition_Augmentor_3_20.pth"))



###################################################################################
# #ask for text description/caption

#caption to text encoding
caption = input('Enter your desired description : ')
seq = []
for word in caption.split():
    seq.append(dataset.rev_vocab[word])
for i in range(len(seq), 100):
    seq.append(0)

seq = th.LongTensor(seq)
seq = seq.cuda()
print(type(seq))
print('\nInput : ', caption)

list_seq = [seq for i in range(16)]
print(len(list_seq))
list_seq = th.stack(list_seq)
list_seq = list_seq.cuda()



embeddings = text_encoder(list_seq)


c_not_hats, mus, sigmas = condition_augmenter(embeddings)


z = th.randn(list_seq.shape[0],
             c_pro_gan.latent_size - c_not_hats.shape[-1]
            ).to(device)

gan_input = th.cat((c_not_hats, z), dim=-1)

alpha = 0.007352941176470588

samples=c_pro_gan.gen(gan_input,
                      current_depth,
                      alpha)

from torchvision.utils import save_image
from torch.nn.functional import upsample
#from train_network import create_grid

img_file = caption + '.png'
samples = (samples / 2) + 0.5
if int(np.power(2, c_pro_gan.depth - current_depth - 1)) > 1:
    samples = upsample(samples, scale_factor=current_depth)


#save image to the disk, the resulting image is <caption>.png
save_image(samples, img_file, nrow=int(np.sqrt(20)))


###################################################################################
# #output the image.
