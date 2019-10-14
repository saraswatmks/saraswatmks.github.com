
---
title: Simple Guide to Packaging a Machine Learning Project using Python PEX 
date: 2019-10-14
tags:
  - machine learning
  - python
  - scikit-learn
excerpt: "Using pex python library to package a python project for deployment."
---

## Introduction

Companies these days are looking for data scientists who can not only build machine learning models but also deploy them. Why? Because, companies are progressing towards data driven product/features development.
Deploying models is not that easy. Specially, if you are a fresher or a data scientist who has been mainly doing dashboarding / data analysis. 

In this tutorial, I will share my knowledge on the first step of deploying a machine learning model i.e. **packing your model code**. Here we'll create a dummy machine learning project and learn to package it.

**PEX (Python executable)** provides an easy way to put together all the code & dependencies and run it as an executable. You no longer have to worry about installing dependencies separately.

*Note:* This tutorial assumes no prior knowledge of how to package a python project.

## Table of Contents

1. Basic Concepts
2. Writing Code
3. Build, Test and Package the Code
4. Deploy


## 1. Basic Concepts

Before we start structuring the project, let's get familiar with some basic concepts in programming: 

**Object Oriented Programming (OOP):** In code packaging, using OOP style (classes & objects) is a widely used methodolgy. Because, it helps in maintaining the code and helps in reducing possible bugs that would get overlooked otherwise.

**Module:** A module is directory (a folder) living in your project which can be imported. A simple way to identify a module is to look for `__init__.py` file. If this file exists in any directory, that directory will be a module.

**Script::** A script is simply a `.py` code file which can be executed. A simple way to identify a script is to look for the entrypoint i.e. `__name__ == '__main__'` line. 

Let's start working now. 

First things first, go to any directory, inside it, create a new directory named `datapackage`. Fill it files shown below:
```
└── datapack
    ├── config
    ├── datapack
    │   ├── __init__.py
    │   ├── docs
    │   └── utils
    ├── requirements.txt
    └── setup.py
```
Inside `datapack`, we will have the following structure:

1. **config:** is a directory which will contain the configuration files.  
2. **datapack:** is a module which will contain the code. It's a good practice to name it same as the project.
  * **docs:** contains files/data to be used by the modules. 
  * **utils** also contains code but mainly helper functions
4. **requirements.txt** contains a list of dependencies to be used.  
5. **setup.py** this file will bundle our project into a package

This is the minimal structure one would need to create a python project. You can always add more directories to organise your project in a way that it is easier to maintain.  

## 2. Writing Code

Now, let's add the code to our `datapack` project. Basically, it would do the following:

* Read data from .csv file
* Do preprocessing (text column mainly)
* Train a model
* Finally, return the predictions.

Now, keep copy-pasting the code and create these scripts at your end. The code is simple and self-explanatory.

`datapack/datapack/utils/preprocess.py` : This script contains code to clean text data. We know this beforehand, because we know that our data contains a column called as *text*.

```python
import re

class PreProcess:
    @staticmethod
    def clean_data(data, stopwords):
        data['text'] = data['text'].apply(lambda x: re.sub(r'\W+',' ', x))
        data['text'] = data['text'].str.lower()
        data['text'] = data['text'].apply(lambda x: ' '.join([y for y in x.split() if y not in stopwords]))
        return data
```

`datapack/datapack/main.py` : This will be the entrypoint of the our project.

```python
from datapack.utils.preprocess import PreProcess
import pandas as pd
import fire
import configparser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB

class TrainPack:

    def __init__(self, config_file):

        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.data = None
        self.stopwords = None

    def read_data(self):
        self.data = pd.read_csv(self.config.get('docs', 'data'))
        self.stopwords = pd.read_csv(self.config.get('docs', 'stopwords'), header=None)[0].tolist()

    def clean_data(self):
        self.data = PreProcess().clean_data(data=self.data, stopwords=self.stopwords)

    def make_features(self):
        text_matrix = CountVectorizer(min_df=2).fit_transform(self.data['text'])
        return text_matrix.todense()

    def train_model(self):
        text_matrix = self.make_features()
        gnb = GaussianNB()
        pred = gnb.fit(text_matrix, self.data['label']).predict(text_matrix)
        return pred

    def run(self):
        self.read_data()
        self.clean_data()
        self.train_model()
        print('finished successfully')

def compute(config_file):
    TrainPack(config_file=config_file).run()

if __name__ == '__main__':
    fire.Fire(compute)
```

`datapack/setup.py` : This file is super important for the successful packaging of the project. It defines the entire setup of the project i.e. which directory are the modules, where the data lives and creates `.whl`, `.tar.gz` extensions for package distribution. 

```python
from codecs import open as codecs_open

from setuptools import setup, find_packages

# Get the long description from the relevant file
with codecs_open('README.md', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='datapack',
    version='0.0.1',
    packages=find_packages(),
    package_data={'datapack': ['docs']},
    url='',
    license='',
    description='Machine learning model',
    long_description=long_description
)
```

`datapack/datapack/utils/__init__.py`  : This file should be empty.

`datapack/datapack/__init__.py`  : Again, this file should be empty.

`datapack/datapack/docs` will contain the sample data. You can download the files from here.

If you have followed all the steps as above, your project should look like this: 
`
datapack
├── Makefile
├── README.md
├── config
│   └── config.ini
├── datapack
│   ├── __init__.py
│   ├── docs
│   │   ├── sample.csv
│   │   └── stopwords.txt
│   ├── main.py
│   └── utils
│       ├── __init__.py
│       └── preprocess.py
├── requirements.txt
└── setup.py
`

Last but not the least, we'll create a `Makefile` which contains commandline code, which we might need to run many times.

`datapack/Makefile` 

```
PYTHON=$(shell which python)

#################################################################################
# Clean	                                                                        #
#################################################################################

clean:
	rm -rf .pytest_cache
	rm -rf datapack.egg-info
	rm -rf target

#################################################################################
# Build pex                                                                     #
#################################################################################

build: clean
	mkdir -p target
	pex . -v --disable-cache -r requirements.txt -R datapack/docs -o target/datapack.pex --python=$(PYTHON)

#################################################################################
# Run pex                                                                     #
#################################################################################

run:
	$(PYTHON) target/datapack.pex -m datapack.main --config_file config/config.ini
```

Basically, it does the following:

* **clean:** Remove all the unnessary residue files which will be created during building the pex.
* **build:** Build the .pex files
* **run:** Runs the pex as a standalone python executable

## 3. Build, Test and Package the code

That's all the code we would need for this project, we are almost in our final stage. To package the project, go the package directory from the terminal and simply run:

`make build`

This would generate some messages on the terminal and finally your project would looks like:

```
datapack
├── Makefile
├── README.md
├── config
│   └── config.ini
├── datapack
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-36.pyc
│   │   └── __init__.cpython-37.pyc
│   ├── docs
│   │   ├── sample.csv
│   │   └── stopwords.txt
│   ├── main.py
│   └── utils
│       ├── __init__.py
│       ├── __pycache__
│       │   └── preprocess.cpython-37.pyc
│       └── preprocess.py
├── datapack.egg-info
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   └── top_level.txt
├── requirements.txt
├── setup.py
└── target
    └── datapack.pex
```

Congrats, you've learned to package your code along with dependencies. `target/datapack.pex` is the packaged code.

Now, let's test if it works. We do:

`make run`

It it works, on your terminal you should see :  `finished successfully`

## 4. Deploy

In deployment, the most commplicated part is to make the project dependencies. You need to make sure that the machine/instance/pod on which you are deploying the model has the same dependencies version. But, no more with pex.

In our case, we have the dependencies along with the code. We just need to make sure about two things:
1. **Python Version:** You need to have the same python version installed to avoid conflicts.
2. **Platform Dependency:** .pex files are platform dependent. That means, you cannot run a .pex file in linux/windows which is build on mac. 

If you've made sure the above two things, just simple copy the .pex file, config file your local to the deployment instance and run the actual `make run` command, which is:

`python3 target/datapack.pex -m datapack.main --config_file config.ini`

## Summary

In this tutorial, we learned about packaging your python project using .pex library for deployment. As mentioned above, pex files are nothing but standalone python interpreters. You could actually run `.pex` files as interpreter on your terminal. In the next tutorial, we'll see how use python 3's zipapp module to package the code.
