from distance.euclidean import Euclidean
from distance.cosine import Cosine
from tensorflow.keras.preprocessing import image
from img_descriptors import extract_img
import os
import numpy as np
import argparse
import cv2

def search_image(img_path, method, distance, top_n):
    metric = None
    if distance == 'cosine':
        metric = Cosine()
    elif distance == 'euclidean':
        metric = Euclidean()

    feature_query = extract_img(img_path, method)

    features_database_path = './data/features_database_2/' + method

    distance_list = []
    name_list = []
    for fi in os.listdir(features_database_path):
        feature_1img_database = np.load(features_database_path + '/' + fi)
        distance_list.append(metric.similarity_metric(feature_query, feature_1img_database))
        name_list.append(fi[:-4])

    sorted_distance_list = np.sort(distance_list)
    argsorted_distance_list = np.argsort(distance_list)

    top_returned_images = []
    for i in argsorted_distance_list[:int(top_n)]:
        path = './data/img/' + name_list[i] + '.jpg'
        top_returned_images.append(path)
    # print(sorted_distance_list)
    # print(argsorted_distance_list)
    print(top_returned_images)
    return top_returned_images


def args_parse():
    parser = argparse.ArgumentParser(description="Retrival Image")
    parser.add_argument('-i', '--input_query_path')
    parser.add_argument('-m', '--method')
    parser.add_argument('-d', '--distance')
    parser.add_argument('-n', '--top_n')

    return vars(parser.parse_args())

def main(args):
  
    img_path = args['input_query_path']
    search_image(img_path, args['method'], args['distance'], args['top_n'])

if __name__== "__main__":
    args = args_parse()
    main(args)