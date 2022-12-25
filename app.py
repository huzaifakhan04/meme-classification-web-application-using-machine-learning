#   Importing all the required libraries/modules.

from flask import Flask, render_template, request
import numpy as np
from nltk.corpus import stopwords
import string
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True    #   To avoid errors while loading truncated images.
from PIL import Image
import re
import pytesseract
import pickle
import os

#   References:
#   •   Ankit, U. (2022, January 6). Image Classification of PCBs and its Web Application (Flask). Medium. https://towardsdatascience.com/image-classification-of-pcbs-and-its-web-application-flask-c2b26039924a
#   •   GeeksforGeeks. (2020, December 26). How to Extract Text from Images with Python? https://www.geeksforgeeks.org/how-to-extract-text-from-images-with-python/
#   •   Neupane, A. (n.d.). GitHub - arpanneupane19/Flask-File-Uploads. GitHub. https://github.com/arpanneupane19/Flask-File-Uploads

application=Flask(__name__, template_folder="templates")    #   Creating an instance of the Flask class.
application.config["UPLOAD_FOLDER"]=r"static\files" #   Specifying the directory for the uploaded images.

#   Function to retrive all the image and text classification models from the stored pickle (.pkl) files.

def get_model():
    global rf_model
    global knc_model
    global etc_model
    global sgd_model
    global mnb_model
    global lr_model
    global vectorizer
    rf_model=pickle.load(open("rf_model.pkl", "rb"))
    knc_model=pickle.load(open("knc_model.pkl", "rb"))
    etc_model=pickle.load(open("etc_model.pkl", "rb"))
    sgd_model=pickle.load(open("sgd_model.pkl", "rb"))
    mnb_model=pickle.load(open("mnb_model.pkl", "rb"))
    lr_model=pickle.load(open("lr_model.pkl", "rb"))
    vectorizer=pickle.load(open("vectorizer.pkl", "rb"))
    print("Image classification models loaded successfully!")

#   Function to open the image file, and pre-process it to the appropriate format.

def load_image(image_path):
    image=Image.open(image_path)
    image=image.convert("L")    #   Converting the image to greyscale.
    image=image.resize((200, 200))  #   Resizing the image to (200 x 200) pixels.
    image=np.array(image)/255.0 #   Converting the image to a numpy array, and normalizing it.
    nx, ny=image.shape  #   Retrieving the dimensions of the image.
    image=image.reshape((1, nx*ny)) #   Reshaping the image to a one-dimensional array.
    return image

#   Function to clean the text extracted from the image.

def clean_text(text):
    text=text.lower()   #   Converting the text to lowercase.
    text=re.sub(r"[^a-z0-9]", " ", text)    #   Removing all the characters except alphabets and numbers.
    text=re.sub(r"\s+", " ", text)  #   Removing all the extra spaces.
    text="".join([character for character in text if character not in string.punctuation])  #   Removing any punctuations from the text.
    text=" ".join([word for word in text.split() if word not in stopwords.words("english")])    #   Removing any stop-words from the text.
    return text

#   Function to predict the sentiment of the image based on the majority voting of all the image classification models.

def get_prediction(image):

    #   Defining the respective classes for the image and text classification models.

    classes=["Neutral", "Positive", "Very Positive", "Negative", "Very Negative"]

    pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    text=pytesseract.image_to_string(image)    #   Extracting the text from the image.
    image=load_image(image) #   Opening, and pre-processing the image.
    text=clean_text(text)   #   Cleaning the text.
    text=vectorizer.transform([text])    #   Transforming the text into a vector.

    #   Predicting the sentiment of the image and extracted text using the image and text classification models.

    rf_answer=rf_model.predict(image)
    knc_answer=knc_model.predict(image)
    etc_answer=etc_model.predict(image)
    sgd_answer=sgd_model.predict(text)
    mnb_answer=mnb_model.predict(text)
    lr_answer=lr_model.predict(text)

    #   Retrieving the respective label for the predicted sentiments.

    rf_answer=classes[rf_answer[0]]
    knc_answer=classes[knc_answer[0]]
    etc_answer=classes[etc_answer[0]]
    sgd_answer=classes[sgd_answer[0]]
    mnb_answer=classes[mnb_answer[0]]
    lr_answer=classes[lr_answer[0]]

    answers=[rf_answer, knc_answer, etc_answer, sgd_answer, mnb_answer, lr_answer]  #   Storing all the predicted sentiments in a list.
    print(answers)
    answer=max(set(answers), key=answers.count) #   Finding the majority voted sentiment.
    return answer

get_model() #   Initialising all the image and text classification models.

#   Function to render the home page of the web application.

@application.route("/", methods=["GET", "POST"])
def home():
    return render_template("home.html")

#   Function to render the result (prediction) page of the web application.

@application.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method=="POST":
        file=request.files["file"]  #   Retrieving the image file from the request.
        filename=file.filename  #   Retrieving the name of the image file.
        file_path=os.path.join(application.config["UPLOAD_FOLDER"], filename)   #   Retrieving the path of the image file.
        file.save(file_path)    #   Saving the image file to the specified directory.
        print(filename)
        prediction=get_prediction(file_path)    #   Predicting the sentiment of the image.
        prediction=prediction.upper()   #   Capitalising the first letter of the string.
        prediction=prediction.replace("_", " ") #   Replacing the underscore with a white-space.
        print(prediction)
    return render_template("predict.html", prediction=prediction, file_path=file_path)

#   Driver function to run the web application.

if __name__=="__main__":
    application.run(debug=True)