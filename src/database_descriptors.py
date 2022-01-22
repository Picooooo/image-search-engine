import img_descriptor
from tensorflow.keras.preprocessing import image
import os
from pathlib import Path
import argparse
import numpy as np

def extract_database(input_database_path, output_path, method):
    feature, feature_path = [], []
    for fi in os.listdir(input_database_path):
        img = image.load_img(os.path.join(input_database_path, fi), target_size=(224, 224))
        feature = img_descriptor.extract_img(img, method)
        feature_path = Path(output_path + "/" + method) / (fi[:-4] + ".npy")
        np.save(feature_path, feature)

def args_parse():
    parser = argparse.ArgumentParser(description="Extraction Database")
    parser.add_argument('-i', '--input_database_path')
    parser.add_argument('-o', '--output_path')
    parser.add_argument('-m', '--method')

    return vars(parser.parse_args())

def main(args):
    extract_database(args['input_database_path'], args['output_path'], args['method'])

if __name__== "__main__":
    args = args_parse()
    main(args)
