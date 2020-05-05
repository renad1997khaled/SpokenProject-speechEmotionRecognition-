import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import linear_model, datasets, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC

from matplotlib import pyplot
from sklearn.base import clone
import warnings
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
warnings.filterwarnings("ignore")
#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X, sampling_rate = librosa.load(file_name,dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))

        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result
#DataFlair - Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'anger',
  '03':'bordom',
  '04':'fear',
  '05':'happines',
  '06':'sadness'
}
#DataFlair - Emotions to observe
observed_emotions=['neutral', 'anger', 'bordom', 'fear','happines','sadness']
#DataFlair - Load the data and extract features for each sound file
train_audio_path='input/train/'
labels=os.listdir(train_audio_path)

#Speach to text Model in Python

def load_data(test_size=0.2):
    x,y=[],[]
    for label in labels:
        print(label)
        waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
        for wav in waves:
            file = train_audio_path + '/' + label + '/' + wav
            feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(label)
    return x,y
#DataFlair - Split the dataset
x,y=load_data(test_size=0.2)
data=train_test_split(np.array(x), y, test_size=0.2, random_state=109)
x_train,x_test,y_train,y_test=data
#Renad Khaled 1151356
#DataFlair - Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))
#DataFlair - Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')
#DataFlair - Initialize the Multi Layer Perceptron Classifier
print("______________________________________________________________________")
print("\nUsing MLPClassifier")
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-05, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
#DataFlair - Train the model
history= model.fit(x_train,y_train)
joblib.dump(history, "model.pkl")
#DataFlair - Predict for the test set
y_pred=model.predict(x_test)
#DataFlair - Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
#DataFlair - Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))
print("______________________________________________________________________")
print("\nUsing GaussianNB")
NB=GaussianNB()
NB.fit(x_train,y_train)
y_pred=NB.predict(x_test)
#DataFlair - Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))
print("______________________________________________________________________")
print("\nUsing SVM")
svm=SVC()
svm.fit(x_train,y_train)
y_pred=svm.predict(x_test)
#DataFlair - Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))
print("______________________________________________________________________")
print("\nUsing BernoulliRBM")
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)
rbm_features_classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = 0.06
rbm.n_iter = 10
# Training RBM-Logistic Pipeline
rbm_features_classifier.fit(x_train, y_train)
# Training the Logistic regression classifier directly on the pixel
raw_pixel_classifier = clone(logistic)
raw_pixel_classifier.C = 100.
raw_pixel_classifier.fit(x_train, y_train)
# Evaluation
# accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
# print("Accuracy: {:.2f}%".format(accuracy*100))
Y_pred = rbm_features_classifier.predict(x_test)
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(y_test, Y_pred)))
accuracy=accuracy_score(y_true=y_test, y_pred=Y_pred)
print("Accuracy using RBM features: {:.2f}%".format(accuracy*100))

Y_pred = raw_pixel_classifier.predict(x_test)
print("\nLogistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(y_test, Y_pred)))
accuracy=accuracy_score(y_true=y_test, y_pred=Y_pred)
print("Accuracy using raw pixel features: {:.2f}%".format(accuracy*100))
print("______________________________________________________________________")
print("\nVotingClassifier(LinearSVC,KNeighborsClassifier,RandomForestClassifier)")
# combine the predictions of several base estimators
clf = VotingClassifier([('lsvc', LinearSVC()),('knn',KNeighborsClassifier()),('rfor', RandomForestClassifier())])
clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)
#DataFlair - Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))
print("______________________________________________________________________")
