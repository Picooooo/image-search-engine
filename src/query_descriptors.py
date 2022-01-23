from img_descriptor import extract_img
import argparse

def args_parse():
    parser = argparse.ArgumentParser(description="Extraction Query")
    parser.add_argument('-i', '--input_image_path')
    parser.add_argument('-m', '--method')

    return vars(parser.parse_args())

def main(args):
    extract_img(args['input_image_path'], args['method'])
