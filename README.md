# Meme Classification Web Application Using Machine Learning:

This repository contains a web application associated with a collection of a few classification algorithms using machine learning in Python to determine the sentiments behind internet memes based on image and text data extracted from 6,992 different internet memes, as part of the final project for the Introduction to Data Science (DS2001) course.

### Dependencies:

* Jupyter Notebook ([install](https://docs.jupyter.org/en/latest/install.html))
* pandas ([install](https://pandas.pydata.org/docs/getting_started/install.html))
* NumPy ([install](https://numpy.org/install/))
* Matplotlib ([install](https://matplotlib.org/stable/users/installing/index.html))
* NLTK ([install](https://www.nltk.org/install.html))
* scikit-learn ([install](https://scikit-learn.org/stable/install.html))
* scikit-image ([install](https://scikit-image.org/docs/stable/install.html))
* Pillow (PIL Fork) ([install](https://pillow.readthedocs.io/en/stable/installation.html))
* Tesseract ([install](https://github.com/tesseract-ocr/tesseract))
* Pytesseract (install - [Anaconda](https://anaconda.org/conda-forge/pytesseract) | install - [PyPI](https://pypi.org/project/pytesseract/))
* Flask ([install](https://flask.palletsprojects.com/en/2.2.x/installation/))

## Introduction:

Classification is defined as the process of recognition, understanding, and grouping of objects and ideas into preset categories (classes). With the help of these pre-categorised training datasets, classification in machine learning programs leverage a wide range of algorithms to classify future datasets into respective and relevant categories (classes). Classification algorithms used in machine learning utilize input training data for the purpose of predicting the likelihood or probability that the data that follows will fall into one of the pre-determined categories.

One of the most common applications of classification algorithms is image and text classification to determine which pre-determined categories certain image and/or text data is the most relevant to. While classification algorithms work for a variety of image and text data, I've trained certain image and text classification models specifically for the classification of internet memes to determine whether a certain meme relays one of five pre-categorised sentiments; neutral, positive, negative, very positive, and very negative. The training [dataset](https://drive.google.com/file/d/1J1SknxcjbjuK0I3OksEleQ7nF53cxdWS/view?usp=share_link) used for the image and text classification models consists of image data from **6,992** different internet memes along with their respective sentiments based on the text data extracted from each of them.

### Classifiers Used (scikit-learn):

* sklearn.ensemble.RandomForestClassifier ([read](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html))
* sklearn.neighbors.KNeighborsClassifier ([read](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html))
* sklearn.ensemble.ExtraTreesClassifier ([read](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html))
* sklearn.linear_model.SGDClassifier ([read](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html))
* sklearn.naive_bayes.MultinomialNB ([read](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html))
* sklearn.linear_model.LogisticRegression ([read](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html))

## Usage:

* ``Meme Classification.ipynb`` — Contains the implementations (scikit-learn) of all trained and tested image and text classification models.
* ``app.py`` — Source code for the web application (Flask) associated with the classification algorithms using machine learning.
* ``test_images`` — Contains the images used for testing the trained image and text classification models.
* ``templates`` — Contains the source codes for the web pages (``home.html`` and ``predict.html``) rendered by the web application (Flask).
* ``static\files`` — Directory used by the web application (Flask) to store the uploaded images into.

## Instructions (Execution):

Firstly, download the training [dataset](https://drive.google.com/file/d/1J1SknxcjbjuK0I3OksEleQ7nF53cxdWS/view?usp=share_link) containing the internet memes to be trained by the classification algorithms using machine learning and extract it into the same directory as the source code files. After that, run all the cells in ``Meme Classification.ipynb``, which will eventually generate the corresponding pickle (``.pkl``) files for each of the trained image and text classification models. Lastly, run ``app.py`` and open the link to the host port. Upload the internet meme (any valid image format) to be tested and its determined sentiment will be displayed accordingly.

#### Note:

The source codes for all the source code files were written entirely for Microsoft Windows and may require certain changes to be run correctly on other operating systems.

---

### References:

* Gong, D. (2022, July 12). *Top 6 Machine Learning Algorithms for Classification.* Medium. https://towardsdatascience.com/top-machine-learning-algorithms-for-classification-2197870ff501
* V, N. (2022, March 15). *Image Classification using Machine Learning.* Analytics Vidhya. https://www.analyticsvidhya.com/blog/2022/01/image-classification-using-machine-learning/
* Anand, A. (2020, June 22). *Basic Machine Learning Cheatsheet using Python [10 Classification & Regression Methods].* DEV Community 👩‍💻👨‍💻. https://dev.to/amananandrai/basic-machine-learning-cheatsheet-using-python-10-classification-regression-methods-9g0
* *1.12. Multiclass and multioutput algorithms.* (n.d.). Scikit-learn. https://scikit-learn.org/stable/modules/multiclass.html
* EliteDataScience. (2022, July 6). *How to Handle Imbalanced Classes in Machine Learning.* https://elitedatascience.com/imbalanced-classes
* Ankit, U. (2022, January 6). *Image Classification of PCBs and its Web Application (Flask).* Medium. https://towardsdatascience.com/image-classification-of-pcbs-and-its-web-application-flask-c2b26039924a
* GeeksforGeeks. (2020, December 26). *How to Extract Text from Images with Python?* https://www.geeksforgeeks.org/how-to-extract-text-from-images-with-python/
* Neupane, A. (n.d.). *GitHub - arpanneupane19/Flask-File-Uploads.* GitHub. https://github.com/arpanneupane19/Flask-File-Uploads
