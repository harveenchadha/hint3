## How can you frame this problem as a Machine Learning problem?

I treat this as a supervised multi class text classification problem. My intention here is to use an End to End DL solution.

I use an uncased base bert in the embedding layers and stacked two fully connected layers on top of it to make it classify text into one of the 21 categories.

### Pros
1. No domain knowledge required to execute the task.
2. Class imbalance can be handled by the network itself.

### Cons
1. Too less data to execute this as a Deep learning problem.
2. If you have a new category tomorrow, you have to retrain the entire model.


## Why do you think your result makes sense?

I have tested on multi-fold validation dataset and getting around 90% accuracy. Also on the test set, the inscope accuracy is around 75%.

I think we can collect more data to improve the model. Either we can generate synthetic data using backtranslation or we can spend effort to spend more data. We can also try a siamese network to check if the two queries are similar.
As the number of categories increase it will become difficult to train the model so we need to have an approach where we define some base queries and then get the queries of the users and check how similar or different they are.


## Reproduce Results

I used a 5 fold stratified strategy to train 5 different models for each fold and later on combine the results by averaging the output probabilities.


### Contents
config folder -> contains the mapping of the labels to int

data -> raw -> contains the raw data

data -> folds.csv -> contains the data in multiple folds

logs -> contains the training tensorboard logs in addition to terminal logs. To check the terminal logs check the training.out file for each fold.

results -> fold wise prediction results of the model

notebooks -> evaluate_oof -> calculates the out of fold accuracy on training and validation set.

notebooks -> test_inference -> checks the inference numbers on the test set.

scripts -> train -> starts the training for a particular fold described in folds.csv present in data folder

scripts -> test -> try to run a single file inference


### How to train?
Go inside scripts folder and use command to start training on fold 0:

```
nohup python train.py 0 &> ../logs/fold_0/training.out &
```
Replicate this process for all the folds from 0 to 4.

### How to test?

Go inside scripts folder and use command to test your custom string:

```
python test.py "Do you give EMI?"
```

### Results
<img width="383" alt="Screenshot 2021-04-07 at 3 12 22 PM" src="https://user-images.githubusercontent.com/30959215/113845997-aa7a1600-97b3-11eb-9ecd-ee15af499e88.png">
