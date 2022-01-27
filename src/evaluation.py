import search
import os
import numpy as np
import argparse
import cv2
from pathlib import Path
from tensorflow.keras.preprocessing import image
import pandas as pd
import matplotlib.pyplot as plt

def read_file(path):
    with open(path) as f:
        lines = f.readlines()
    contents = [x.strip() for x in lines]
    return contents  

def compute_AP(returned, gt):
  total = len(gt)
  precision = 0
  number = 0

  compared_result = []
  for i in range(len(returned)):
    if returned[i] in gt:
      compared_result.append(1)
    else:
      compared_result.append(0)
  for i, value in enumerate(compared_result):
    if value == 1:
      precision += (number+1)/(i+1)
      number += 1
  return precision / total

def evaluation(method, distance, top_n):
  df = pd.DataFrame({'Query': [], 'Average Precision': []})
  labels = ['good']
  mAP = 0
  AP = []
  for query_file_path in sorted(Path('./data/ground_truth').glob('*_query.txt')):
      query_image_name = read_file(query_file_path)[0].split()[0].replace('oxc1_', '')

      print("QUERY:", query_file_path, query_image_name+'.jpg')

      query_coords = read_file(query_file_path)[0].split()[1:]
      query_coords = [float(i) for i in query_coords]

      query_image_path = './data/img/' + query_image_name + '.jpg'
      # img = cv2.imread(query_image_path)
      # cropped_img = img[round(query_coords[1]):round(query_coords[3]),
      #               round(query_coords[0]):round(query_coords[2])]

      returned_images = search.search_image(query_image_path, method, distance, top_n)
      gt_images = []
      for label in labels:
        gt_images.extend(read_file(str(query_file_path).replace('query', label)))
      ap = compute_AP(returned_images, gt_images)
      print('AP:', ap)
      AP.append(ap)
      df = df.append({'Query': query_image_name, 'Average Precision': ap}, ignore_index=True)

      print('---------------------------------------')
  df.plot(x='Query', y='Average Precision', kind='bar')
  plt.show()

  mAP = sum(AP)/len(AP)
  print('mAP:', mAP)
  return mAP

def args_parse():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument('-m', '--method')
    parser.add_argument('-d', '--distance')
    parser.add_argument('-n', '--top_n')

    return vars(parser.parse_args())

def main(args):
    evaluation(args['method'], args['distance'], args['top_n'])

if __name__== "__main__":
    args = args_parse()
    main(args)