import pandas as pd
import numpy as np
from transformers import BertTokenizer
import tensorflow as tf
from transformers import TFAutoModel
from tqdm import tqdm
from sklearn.metrics import classification_report
import os
import json
import sys
from sklearn.metrics import f1_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

seq_len = 128
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def set_seed(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    
def ret_token( phrase):
    tokens = tokenizer.encode_plus(phrase,
                                   max_length = seq_len,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=True,
                                   return_tensors='tf', return_token_type_ids=False )
    
    return {'input_ids':tf.cast(tokens['input_ids'], tf.float64), 'attention_mask':tf.cast(tokens['attention_mask'], tf.float64)}

def get_prediction(data):
    _predicted_probs = []
    #for item in data:
    ret = ret_token(data.lower())
    
    
    for i in tqdm(range(0,5)):
        my_model = tf.keras.models.load_model('../checkpoints/fold_'+str(i)+'/best_model.h5')
        probs = my_model.predict(ret)
        _predicted_probs.append(probs)
    
    _predicted_probs = [item[0] for item in _predicted_probs]
    mean_probs = np.mean(_predicted_probs, axis=0)
#     print(_predicted_probs)
#     print(mean_probs)
    
    class_prob = np.argmax(mean_probs)
    confidence = mean_probs[class_prob]
    
    with open('../config/labels.json') as json_file:
        label_mapping = json.load(json_file)
        
    label_mapping = dict([(value, key) for key, value in label_mapping.items()])
    
    class_pred = label_mapping[class_prob]
    if confidence < 0.1:
        print(class_pred, confidence)
        return "No Node Detected",0
    
    return class_pred, confidence


if __name__ == "__main__":
    sent = sys.argv[1]
    class_pred, conf = get_prediction(sent)
    print("Intent Predicted:", class_pred)
    if conf!=0:
        print("Confidence:",conf)