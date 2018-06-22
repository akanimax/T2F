""" script to process and save the text annotations for the lfw dataset
    from the Face2Text dataset.
"""

import argparse


def parse_arguments():
    """
    command line argument parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_annot_file", action="store", type=str,
                        default="../data/LFW/Face2Text/face2text_v0.1/clean.json",
                        help="path to the json annotations file (Face2Text data)")

    parser.add_argument("--out_file", action="store", type=str,
                        default="processed_annotations/processed_text.pkl",
                        help="path to store the output pickle file")

    args = parser.parse_args()
    return args


def main(args):
    """
    Main function of the script
    :param args: parsed command line arguments
    :return: None
    """
    import data_processing.TextExtractor as te

    # read the annotations:
    images, descs = te.read_annotations(args.json_annot_file)

    # preprocess the descriptions
    print("basic preprocessing of the data ...")
    descs = te.basic_preprocess(descs)

    # tokenize and obtain vocabularies for the text
    print("tokenization and transformation ...")
    vocab, rev_vocab, data = te.tokenize(descs, te.frequency_count(descs))

    # create the object for saving:
    data_obj = {
        'images': images,
        'vocab': vocab,
        'rev_vocab': rev_vocab,
        'data': data
    }

    # save the data_obj at the output file
    print("saving the file ...")
    te.save_pickle(data_obj, args.out_file)

    print("processed data file has been saved at:", args.out_file)


if __name__ == '__main__':
    main(parse_arguments())
