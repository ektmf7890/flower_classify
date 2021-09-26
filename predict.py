import json
import argparse
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_hub as hub

from preprocess import process_image

parser = argparse.ArgumentParser()
parser.add_argument('img_path', type=str, help='Path to image file.')
parser.add_argument('model_path', type=str, help='Path to model file.')
parser.add_argument('--top_k', type=int, help='Return the top KKK most likely classes:')
parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names:')
args = parser.parse_args()

def predict(img_path, model_path, k):
    if not k:
        k = 5
    
    class_names = None
    if args.category_names and os.path.isfile(args.category_names):
        with open(args.category_names, 'r') as f:
          class_names = json.load(f)
    
    if model_path and os.path.isfile(model_path):
        model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False) 
#         model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    else: return
    
    if img_path and os.path.isfile(img_path):
        img = process_image(img_path)
        img = np.expand_dims(img, 0) # (224, 224, 3) -> (1, 224, 224, 3)
    else: return

    predictions = model.predict(img)[0]
  
    indices = np.argsort(predictions)[-k:][::-1]
    topk_probs = predictions[indices]
    
    if class_names:
        topk_classes = [class_names.get(str(i+1)) for i in indices]
    else:
        topk_classes = [i + 1 for i in indices]
        
    for class_, prob in zip(topk_classes, topk_probs):
        print('%s: %.4f' %(class_, prob))
    
    
if __name__ == "__main__":      
    tf.keras.backend.clear_session()
    predict(args.img_path, args.model_path, args.top_k)
    