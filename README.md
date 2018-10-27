# Wine Classifier
Predicts wines one would most likely drink using Tensorflow's Wide and Deep Learning on [Wine Enthusiast](https://www.kaggle.com/zynicide/wine-reviews)'s reviews on 130k wines. 
Current dataset is based on a single reviewer (points >= 90 are labeled as "drinkable"). 

Predictors:
* grape variety
* country of origin
* price
* review text

Review text is modeled as both _dense embeddings_ (*deep*) and _sparse bag of words_ (*wide*). The latter captures exceptions that considerably boost precision and recall.

Classifier may be extended into a recommender to predict for individual preferences if dataset includes multiple reviewers.

### Instructions
Run program.py which will call data.py (clean and preprocess data). Helper functions in utils.py

### Results
* Precision: 0.903, Recall: 0.877 (variety + country + price + BOW)
* Precision: 0.908, Recall: 0.874 (variety + price + BOW)
* Precision: 0.913, Recall: 0.843 (variety + BOW)
* Precision: 0.558, Recall: 0.067 (variety + country)

### Builds
Tensorflow 1.9
Keras 2.2.2

## Acknowledgments
This work is inspired by Sara Robinson's [blog post](https://medium.com/tensorflow/predicting-the-price-of-wine-with-the-keras-functional-api-and-tensorflow-a95d1c2c1b03) on predicting wine prices
