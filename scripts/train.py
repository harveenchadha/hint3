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
  
    
seq_len = 128
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def set_seed(SEED):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
def get_id_masks(tokenizer, data):
    ids = []
    masks = []
    for phrase in tqdm(data):
        tokens = tokenizer.encode_plus(phrase,
                                       max_length = seq_len,
                                       truncation=True,
                                       padding='max_length',
                                       add_special_tokens=True,
                                       return_tensors='tf')
        ids.append( tokens['input_ids'][0] )
        masks.append( tokens['attention_mask'][0] )
    return ids, masks

def map_func(X_ids, X_masks, label):
    return {'input_ids': X_ids, 'attention_mask':X_masks}, label

def train_and_valid_data(valid_fold):
    with open('../config/labels.json') as json_file:
        label_mapping = json.load(json_file)
    
    folds = pd.read_csv('../data/folds.csv')
    folds.label = folds.label.map(label_mapping)
    train_data = folds[folds.fold != valid_fold].reset_index(drop=True)#.drop(columns=['fold'])
    valid_data = folds[folds.fold == valid_fold].reset_index(drop=True)#.drop(columns=['fold'])
    train_data = pd.get_dummies(train_data, columns=['label'])
    valid_data = pd.get_dummies(valid_data, columns=['label'])
    train_data['sentence']=train_data['sentence'].str.lower()
    valid_data['sentence']=valid_data['sentence'].str.lower()
    label_train=train_data.iloc[:,3:]
    label_valid = valid_data.iloc[:,3:]
    
    return train_data, valid_data, label_train, label_valid

def prepare_data_for_bert(train_data, valid_data):

    
    
    X_train_ids, X_train_masks = get_id_masks(tokenizer, train_data['sentence'])
    X_valid_ids, X_valid_masks = get_id_masks(tokenizer, valid_data['sentence'])
    
    dataset_train = tf.data.Dataset.from_tensor_slices((X_train_ids, X_train_masks, label_train))
    dataset_valid = tf.data.Dataset.from_tensor_slices((X_valid_ids, X_valid_masks, label_valid))
    
    dataset_train = dataset_train.map(map_func).shuffle(1000).batch(8, drop_remainder=True)
    dataset_valid = dataset_valid.map(map_func).shuffle(1000).batch(8, drop_remainder=True)
    
    return tokenizer, dataset_train, dataset_valid

def define_model(train_data):
    bert = TFAutoModel.from_pretrained('bert-base-uncased')
    
    _input_ids = tf.keras.layers.Input(shape=(seq_len,), name='input_ids', dtype='int32')
    _input_masks = tf.keras.layers.Input(shape=(seq_len, ), name='attention_mask', dtype='int32')

    embeddings = bert.bert(_input_ids, _input_masks)[1]

    x = tf.keras.layers.Dense(1024, activation='relu')(embeddings)
    drop = tf.keras.layers.Dropout(0.5)(x)
    y = tf.keras.layers.Dense(21, activation='softmax')(drop)


    model = tf.keras.Model(inputs= [_input_ids, _input_masks], outputs=y) 
    print(model.summary())
    return model

def train_model(valid_fold, dataset_train, dataset_valid):
    mc = tf.keras.callbacks.ModelCheckpoint('../checkpoints/fold_'+str(valid_fold)+'/best_model.h5',verbose=1, save_best_only=True)
    lm = tf.keras.callbacks.ModelCheckpoint('../checkpoints/fold_'+str(valid_fold)+'/last_model.h5',verbose=1, save_best_only=False)

    plat = tf.keras.callbacks.ReduceLROnPlateau(patience=2, verbose=1)
    es = tf.keras.callbacks.EarlyStopping(verbose=1, patience=5)
    tb = tf.keras.callbacks.TensorBoard(log_dir = '../logs/fold_'+str(valid_fold))
    
    
    optimizer = tf.keras.optimizers.Adam(lr = 1e-4)
    loss = tf.keras.losses.CategoricalCrossentropy()
    acc = tf.keras.metrics.CategoricalAccuracy('accuracy')
    model.compile(optimizer,loss = loss, metrics = [acc])

    model.fit(dataset_train, epochs=25, validation_data=dataset_valid, callbacks=[mc, plat, es, lm, tb])

    
def ret_token( phrase):
    tokens = tokenizer.encode_plus(phrase,
                                   max_length = seq_len,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=True,
                                   return_tensors='tf', return_token_type_ids=False )
    
    return {'input_ids':tf.cast(tokens['input_ids'], tf.float64), 'attention_mask':tf.cast(tokens['attention_mask'], tf.float64)}

def get_prediction(my_model, data):
    _predicted_probs = []
    for item in tqdm(data['sentence']):
        ret = ret_token(item.lower())
        probs = my_model.predict(ret)
        _predicted_probs.append(probs)
    return _predicted_probs

def get_full_data_preds(my_model, data):
    _preds = get_prediction(my_model, data)
    _preds = [item[0] for item in _preds]
    _preds_df = pd.DataFrame(_preds)
    return pd.concat([data, _preds_df], axis=1)
    
def execute_inference(train_data, valid_data):
    my_model = tf.keras.models.load_model('../checkpoints/fold_'+str(valid_fold)+'/best_model.h5')
    train_preds = get_full_data_preds(my_model, train_data)
    valid_preds= get_full_data_preds(my_model, valid_data)
    os.makedirs('../results/fold_'+str(valid_fold), exist_ok=True)
    
    train_preds.to_csv('../results/fold_'+str(valid_fold)+'/train_preds.csv', index=False)
    valid_preds.to_csv('../results/fold_'+str(valid_fold)+'/valid_preds.csv', index=False)


if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)
    
    valid_fold = int(sys.argv[1])
    
    df = pd.read_csv('../data/raw/sofmattress_train.csv')
    
    train_data, valid_data, label_train, label_valid = train_and_valid_data(valid_fold)
    
    tokenizer, dataset_train, dataset_valid = prepare_data_for_bert(train_data, valid_data)
    
    model = define_model(train_data)
    
    os.makedirs('../checkpoints/fold_'+str(valid_fold), exist_ok=True)
    os.makedirs('../logs/fold_'+str(valid_fold), exist_ok=True)
    
    print("Training Started")
    train_model(valid_fold, dataset_train, dataset_valid)
    print("Training Ended")
    
    execute_inference(train_data, valid_data)
    
    print("Result saved for training and validation for fold "+str(valid_fold))

    