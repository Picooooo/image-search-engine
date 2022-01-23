from distance.euclidean import Euclidean
from distance.cosine import Cosine
from img_descriptor import extract_img
import os
import numpy as np
import argparse
import cv2

def search_image(img, method, distance):
    metric = None
    if distance == 'cosine':
        metric = Cosine()
    elif distance == 'euclidean':
        metric = Euclidean()

    feature_query = extract_img(img, method)

    features_database_path = './data/features_database/' + method

    distance_list = []
    name_list = []
    for fi in os.listdir(features_database_path):
        feature_1img_database = np.load(features_database_path + '/' + fi)
        distance_list.append(metric.similarity_metric(feature_query, feature_1img_database))
        name_list.append(fi[:-4])

    sorted_distance_list = np.sort(distance_list)
    argsorted_distance_list = np.argsort(distance_list)

    top_img_file = []
    for i in argsorted_distance_list[:5]:
        path = './data/img/' + name_list[i] + '.jpg'
        top_img_file.append(path)
        print(path)


def args_parse():
    parser = argparse.ArgumentParser(description="Retrival Image")
    parser.add_argument('-i', '--input_query_path')
    parser.add_argument('-m', '--method')
    parser.add_argument('-d', '--distance')

    return vars(parser.parse_args())

def main(args):
    img = cv2.imread(args['input_query_path'])
    search_image(img, args['method'], args['distance'])

if __name__== "__main__":
    args = args_parse()
    main(args)