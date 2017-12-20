#SMSSpamFilter

## Introduction
Is a project that allows you to discern if a short message (essentially an SMS) is a spam or not. This is possible thanks to a binary classification model obtained by applying machine algorithms.

These algorithms are detected using a Python automated machine learning tool that optimizes machine learning pipelines using genetic programming ([TPOT](https://github.com/rhiever/tpot)).
The Pipelines are [scikit-learn Pipelines](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), so, if you are familiarized with scikit-learn you are good to go.

Although *SMSSpamFilter's* main task is to verify if a SMS is a spam, it could be applied to another binary classification task with small changes.

# Why TPOT
The data mining process is an iterative task that involves several cycles in order to achieve good results.
TPOT help you to perform automatically hundreds or thousands of combinations using advances techniques for features selection, preprocessing and construction, also, evaluating different machine learning algorithms with the goal of detecting the Pipeline that gives the best results from the accuracy point of view.

There are other automatic machine learning libraries, like [AutoML](https://github.com/automl/auto-sklearn), that could be evaluated in the feature.

# How SMSSpamFilter works
*SMSSpamFilter* has two main modules, `spamdetector.pipeline_selector` and `spamdetector.model_builder`, the first is in charge of selecting/generating the best Pipeline and the second has the responsibility of training a model using the selected Pipeline.

The current implementation (refer to the `factory.get_pipeline()`) has a pretty good **Accuracy Score: 0.989247311828**, but depends as well about the provided data, the training and testing data.

# Dataset format
The way *SMSSpamFilter* gets the data is through command line parameter. The parameter value is the path to the file that contains the data and must obey the following format:

```
ham	this is an example of a document that is not spam
spam	I am a spam, please do not delet e m e, buy, click here, viagra
```

The category or label of the example is `ham` or `spam` and **must be separated from the document data by a `tab`**, hence, the format of the data is a tab separated file.

# Requirements
**It is highly recommended to execute everything within a [Python virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/)**
The computer used to run this project should have installed Python 3.5 and Make

# Generating the Pipeline
In other to obtain another Pipeline different from the one already provided by `factory.get_pipeline()`, execute:

```
make select_pipeline dataset_path=PATH/TO/THE/TAB/SEPARATED/DATASET
```

After the execution, a file with the selected Pipeline will be written to the configuration.BEST_PIPELINE_PYTHON_FILENAME, please check the `configuration` module.

The Pipeline that comes with *SMSSpamFilter* has been obtained after training/fitting the classifier with 50 generations
and 50 population size, using all the available CPU on my computer, a MacBook Pro with a Corei7, 12GB of RAM.

In my case, it took around 20 hours and this is how the output looks like:
```
Warning: xgboost.XGBClassifier is not available and will not be used by TPOT.
Generation 1 - Current best internal CV score: 0.9854472375828436
Generation 2 - Current best internal CV score: 0.9854472375828436
Generation 3 - Current best internal CV score: 0.9854472375828436
Generation 4 - Current best internal CV score: 0.9854472375828436
Generation 5 - Current best internal CV score: 0.9862438517244824
Generation 6 - Current best internal CV score: 0.9862438517244824
Generation 7 - Current best internal CV score: 0.9866430533212889
Generation 8 - Current best internal CV score: 0.9866430533212889
Generation 9 - Current best internal CV score: 0.9878388690597341
Generation 10 - Current best internal CV score: 0.9878388690597341
Generation 11 - Current best internal CV score: 0.9878388690597341
Generation 12 - Current best internal CV score: 0.9878388690597341
Generation 13 - Current best internal CV score: 0.9878388690597341
Generation 14 - Current best internal CV score: 0.9878392662744643
Generation 15 - Current best internal CV score: 0.9878392662744643
Generation 16 - Current best internal CV score: 0.9878392662744643
Generation 17 - Current best internal CV score: 0.9878392662744643
Generation 18 - Current best internal CV score: 0.9878398620965598
Generation 19 - Current best internal CV score: 0.9878398620965598
Generation 20 - Current best internal CV score: 0.9878398620965598
Generation 21 - Current best internal CV score: 0.9878398620965598
Generation 22 - Current best internal CV score: 0.9878398620965598
Generation 23 - Current best internal CV score: 0.9878398620965598
Generation 24 - Current best internal CV score: 0.9878398620965598
Generation 25 - Current best internal CV score: 0.9878398620965598
Generation 26 - Current best internal CV score: 0.9878398620965598
Generation 27 - Current best internal CV score: 0.9878398620965598
Generation 28 - Current best internal CV score: 0.9878398620965598
Generation 29 - Current best internal CV score: 0.9878398620965598
Generation 30 - Current best internal CV score: 0.9878398620965598
Generation 31 - Current best internal CV score: 0.9878398620965598
Generation 32 - Current best internal CV score: 0.9878398620965598
Generation 33 - Current best internal CV score: 0.9878398620965598
Generation 34 - Current best internal CV score: 0.9878398620965598
Generation 35 - Current best internal CV score: 0.9878398620965598
Generation 36 - Current best internal CV score: 0.9878398620965598
Generation 37 - Current best internal CV score: 0.9878398620965598
Generation 38 - Current best internal CV score: 0.9878398620965598
Generation 39 - Current best internal CV score: 0.9878398620965598
Generation 40 - Current best internal CV score: 0.9878398620965598
Generation 41 - Current best internal CV score: 0.9878398620965598
Generation 42 - Current best internal CV score: 0.9878398620965598
Generation 43 - Current best internal CV score: 0.9878398620965598
Generation 44 - Current best internal CV score: 0.9878398620965598
Generation 45 - Current best internal CV score: 0.9878398620965598
Generation 46 - Current best internal CV score: 0.9878398620965598
Generation 47 - Current best internal CV score: 0.9878398620965598
Generation 48 - Current best internal CV score: 0.9878398620965598
Generation 49 - Current best internal CV score: 0.9878398620965598
Generation 50 - Current best internal CV score: 0.9878398620965598

Best pipeline: BernoulliNB(RFE(input_matrix, criterion=entropy, max_features=0.15, n_estimators=100, step=0.35), alpha=0.01, fit_prior=True)
```

# Generating the classification model
After generating the Pipeline, and in case you want to change the Pipeline provided by *SMSSpamFilter*, go the file configured in `configuration.BEST_PIPELINE_PYTHON_FILENAME`, get to generated code and substitute the implementation of the `factory.get_pipeline()`

Then execute the following, either you want to use the provided Pipeline our your own:

```
make create_model dataset_path=PATH/TO/THE/TAB/SEPARATED/DATASET
```

After executing this, you should get an output like the following:

```
INFO:root:Getting features from dataset...
INFO:root:Vectorizing examples...
INFO:root:Splitting dataset...
INFO:root:Transforming and fitting pipeline...
INFO:root:Reporting...
INFO:root:Accuracy score: 0.9946236559139785
INFO:root:             precision    recall  f1-score   support

        ham       0.99      1.00      1.00       478
       spam       1.00      0.96      0.98        80

avg / total       0.99      0.99      0.99       558

INFO:root:  Confusion Matrix
INFO:root:       ham spam
INFO:root:  ham   478 0
INFO:root:  spam   3 77
```

**OPTIONAL**
There is a possibility to perform oversampling in case your dataset is not balanced, hence, could be biased towards one category. Then change the configuration of `configuration.PERFORM_OVERSAMPLING` to `True` and voilà.

*The test dataset portion will remain unbalanced*

# Testing
```
make test
```

easy enough :)

**NOTE**: The first time execution could take a while due dependencies installation

# Contributing
Send me a pull request and I will be glad to review and accept it
