import librosa
import soundfile
import os, pickle
import numpy as np
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn import datasets,linear_model
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.base import clone
import warnings
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
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
def load_data(test_size=0.2):
    x,y=[],[]
    for label in labels:
        print(label)
        waves = [f for f in os.listdir(train_audio_path + '/' + label) if f.endswith('.wav')]
        for wav in waves:
            file = train_audio_path + '/' + label + '/' + wav
            feature = extract_feature(file, mfcc=True, chroma=False, mel=False)
            x.append(feature)
            y.append(label)
    return x,y
def load_data2(test_size=1):
    x,y=[],[]
    waves = [f for f in os.listdir(test_audio_path+'/' ) if f.endswith('.wav')]
    for wav in waves:
        file =test_audio_path + '/' + wav
        feature = extract_feature(file, mfcc=True, chroma=False, mel=False)
        x.append(feature)
        y.append(wav)
    return x,y
def CreateFile(y_pred):
    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
    name=timestr+".txt"
    file1 = open(name, "a")  # append mode
    i = 1
    for m in y_pred:
        print(i, m)
        i = i + 1
        if m=="anger":
            file1.write("1\n")
        elif m=="bordom":
            file1.write("2\n")
        elif m=="fear":
            file1.write("3\n")
        elif m=="happines":
            file1.write("4\n")
        elif m=="sadness":
            file1.write("5\n")
        elif m=="neutral":
            file1.write("6\n")
    file1.close()
    file1 = open(name, "r")
    print("Output of Readlines after appending")
    print (file1.readlines())
    file1.close()
    return
def predict(audio):
    prob=loaded_model.predict(audio)
    return prob
classes={
  '1':'anger',
  '2':'bordom',
  '3':'fear',
  '4':'happines',
  '5':'sadness',
  '6':'neutral'
}
test_audio_path='input/wav_tests'
#DataFlair - Load the data and extract features for each sound file
train_audio_path='input/train/'
labels=os.listdir(train_audio_path)
#Speach to text Model in Python
warnings.filterwarnings("ignore")
#DataFlair - Split the dataset
x,y=load_data(test_size=0.1)
data=train_test_split(np.array(x), y, test_size=0.1, random_state=109)
x_train,x_test,y_train,y_test=data
#Renad Khaled 1151356
#DataFlair - Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))
#DataFlair - Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')
#DataFlair - Initialize the Multi Layer Perceptron Classifier
print("______________________________________________________________________")
print("\nUsing MLPClassifier")
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-05, hidden_layer_sizes=(300,200,50), learning_rate='adaptive', max_iter=500)
#DataFlair - Train the model
history= model.fit(x_train,y_train)
# save model and architecture to single file
joblib.dump(history, "model.h5")
print("Saved model to disk")
#DataFlair - Predict for the test set
y_pred=model.predict(x_test)
#DataFlair - Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print(metrics.classification_report(y_test, y_pred))
#DataFlair - Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))
#load the model to test the model again
loaded_model = joblib.load('model.h5')
x,y=load_data2(test_size=1)
y_pred=predict(x)
#save the prediction according to their number to a txt file
CreateFile(y_pred)
print("______________________________________________________________________")
print("\nUsing GaussianNB")
NB=GaussianNB()
NB.fit(x_train,y_train)
y_pred=NB.predict(x_test)
#DataFlair - Calculate the accuracy of our model
print(metrics.classification_report(y_test, y_pred))
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
print(metrics.classification_report(y_test, y_pred))
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))
print("______________________________________________________________________")
