""" Module containing basic data readers and extractors """

import json
import re
import pickle


def read_annotations(file_path):
    """
    read the json annotations (textual descriptions)
    :param file_path: path of the json file
    :return: annos => read annotations
    """
    with open(file_path, "r") as json_desc:
        annos = json.load(json_desc)

    images, descriptions = [], []  # initialize to empty lists
    # convert all the file's extensions to .jpg
    for anno in annos:
        name, _ = anno['image'].split('.')
        new_name = name + '.jpg'
        anno['image'] = new_name

        # extract only the image_id and the descriptions in a list
        for desc in anno['descriptions']:
            images.append(anno['image'])
            descriptions.append(desc['text'].lower())

    # check if their lengths match:
    assert len(images) == len(descriptions), "something messed up while reading data ..."

    # return the read annos
    return images, descriptions


def basic_preprocess(descriptions):
    """
    basic preprocessing on the input data
    :param descriptions: list[strings]
    :return: dat => list[lists[string]]
    """

    # insert space before all the special characters
    db_desc = []
    for desc in descriptions:
        punctuations = re.sub(r"([^a-zA-Z])", r" \1 ", desc)
        excess_space = re.sub('\s{2,}', ' ', punctuations)
        db_desc.append(excess_space)

    return db_desc


def frequency_count(text_data):
    """
    count the frequency of each word in data
    :param text_data: list[string]
    :return: freq_cnt => {word -> freq}
    """
    text_data = list(map(lambda x: x.split(), text_data))
    # generate the vocabulary
    total_word_list = []
    for line in text_data:
        total_word_list.extend(line)

    vocabulary = set(total_word_list)

    freq_count = dict(map(lambda x: (x, 0), vocabulary))

    # count the frequencies of the words
    for line in text_data:
        for word in line:
            freq_count[word] += 1

    # return the frequency counts
    return freq_count


def tokenize(text_data, freq_counts, vocab_size=None):
    """
    tokenize the text_data using the freq_counts
    :param text_data: list[string]
    :param freq_counts: {word -> freq}
    :param vocab_size: size of the truncated vocabulary
    :return: (rev_vocab, trunc_vocab, transformed_data
                => reverse vocabulary, truncated vocabulary, numeric sequences)
    """
    # split the text_data into word lists
    text_data = list(map(lambda x: x.split(), text_data))

    # truncate the vocabulary:
    vocab_size = len(freq_counts) if vocab_size is None else vocab_size

    trunc_vocab = dict(sorted(freq_counts.items(), key=lambda x: x[1], reverse=True)[:vocab_size])
    trunc_vocab = dict(enumerate(trunc_vocab.keys(), start=2))

    # add <unk> and <pad> tokens:
    trunc_vocab[1] = "<unk>"
    trunc_vocab[0] = "<pad>"

    # compute reverse trunc_vocab
    rev_trunc_vocab = dict(list(map(lambda x: (x[1], x[0]), trunc_vocab.items())))

    # transform the sentences:
    transformed_data = []  # initialize to empty list
    for sentence in text_data:
        transformed_sentence = []
        for word in sentence:
            numeric_code = rev_trunc_vocab[word] \
                if word in rev_trunc_vocab else rev_trunc_vocab["<unk>"]
            transformed_sentence.append(numeric_code)

        transformed_data.append(transformed_sentence)

    # return the truncated vocabulary and transformed sentences:
    return trunc_vocab, rev_trunc_vocab, transformed_data


def save_pickle(obj, file_name):
    """
    save the given data obj as a pickle file
    :param obj: python data object
    :param file_name: path of the output file
    :return: None (writes file to disk)
    """
    with open(file_name, 'wb') as dumper:
        pickle.dump(obj, dumper, pickle.HIGHEST_PROTOCOL)


def load_pickle(file_name):
    """
    load a pickle object from the given pickle file
    :param file_name: path to the pickle file
    :return: obj => read pickle object
    """
    with open(file_name, "rb") as pick:
        obj = pickle.load(pick)

    return obj
