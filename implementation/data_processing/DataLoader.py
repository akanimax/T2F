""" Module for the data loading pipeline for the model to train """

import PIL
import torch as th
import os

from torch.utils.data import Dataset


class Face2TextDataset(Dataset):
    """ PyTorch Dataset wrapper around the Face2Text dataset """

    def __load_data(self):
        """
        private helper for loading the data
        :return: data => dict of data objs
        """
        from data_processing.TextExtractor import load_pickle
        data = load_pickle(self.pickle_file_path)

        return data

    def __init__(self, pro_pick_file, img_dir, img_transform=None, captions_len=100):
        """
        constructor of the class
        :param pro_pick_file: processed pickle file
        :param img_dir: path to the images directory
        :param img_transform: torch_vision transform to apply
        :param captions_len: maximum length of the generated captions
        """

        # create state:
        self.base_path = img_dir
        self.pickle_file_path = pro_pick_file
        self.transform = img_transform
        self.max_caption_len = captions_len

        data_obj = self.__load_data()

        # extract all the data
        self.text_data = data_obj['data']
        self.rev_vocab = data_obj['rev_vocab']
        self.vocab = data_obj['vocab']
        self.images = data_obj['images']
        self.vocab_size = len(self.vocab)

    def __len__(self):
        """
        obtain the length of the data-items
        :return: len => length
        """
        return len(self.images)

    def __getitem__(self, ix):
        """
        code to obtain a specific item at the given index
        :param ix: index for element query
        :return: (caption, img) => caption and the image
        """

        # read the image at the given index
        img_file_path = os.path.join(self.base_path, self.images[ix])
        img = PIL.Image.open(img_file_path)

        # transform the image if required
        if self.transform is not None:
            img = self.transform(img)

        # get the encoded caption:
        caption = self.text_data[ix]

        # pad or truncate the caption length:
        if len(caption) < self.max_caption_len:
            while len(caption) != self.max_caption_len:
                caption.append(self.rev_vocab["<pad>"])

        elif len(caption) > self.max_caption_len:
            caption = caption[: self.max_caption_len]

        caption = th.tensor(caption, dtype=th.long)

        # return the data element
        return caption, img

    def get_english_caption(self, sent):
        """
        obtain the english words list for the given numeric sentence
        :param sent: numeric id sentence
        :return: sent => list[String]
        """
        return list(map(lambda x: self.vocab[x], sent.numpy()))


class RawTextFace2TextDataset(Dataset):
    """ PyTorch Dataset wrapper around the Face2Text dataset
        Raw text version
    """

    def __load_data(self):
        """
        private helper for loading the annotations and file names from the annotations file
        :return: images, descs => images and descriptions
        """
        from data_processing.TextExtractor import read_annotations, basic_preprocess
        images, descs = read_annotations(self.annots_file_path)
        # preprocess the descriptions:
        descs = basic_preprocess(descs)

        return images, descs

    def __init__(self, annots_file, img_dir, img_transform=None):
        """
        constructor of the class
        :param annots_file: annotations file
        :param img_dir: path to the images directory
        :param img_transform: torch_vision transform to apply
        """

        # create state:
        self.base_path = img_dir
        self.annots_file_path = annots_file
        self.transform = img_transform

        self.images, self.descs = self.__load_data()

        # extract all the data

    def __len__(self):
        """
        obtain the length of the data-items
        :return: len => length
        """
        return len(self.images)

    def __getitem__(self, ix):
        """
        code to obtain a specific item at the given index
        :param ix: index for element query
        :return: (caption, img) => caption and the image
        """

        # read the image at the given index
        img_file_path = os.path.join(self.base_path, self.images[ix])
        img = PIL.Image.open(img_file_path)

        # transform the image if required
        if self.transform is not None:
            img = self.transform(img)

        # get the raw_text caption:
        caption = self.descs[ix]

        # return the data element
        return caption, img


def get_transform(new_size=None):
    """
    obtain the image transforms required for the input data
    :param new_size: size of the resized images
    :return: image_transform => transform object from TorchVision
    """
    from torchvision.transforms import ToTensor, Normalize, Compose, Resize

    if new_size is not None:
        image_transform = Compose([
            Resize(new_size),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    else:
        image_transform = Compose([
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    return image_transform


def get_data_loader(dataset, batch_size, num_workers):
    """
    generate the data_loader from the given dataset
    :param dataset: F2T dataset
    :param batch_size: batch size of the data
    :param num_workers: num of parallel readers
    :return: dl => dataloader for the dataset
    """
    from torch.utils.data import DataLoader

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return dl
