import librosa
import IPython.display as ipd
import numpy as np
from keras.models import model_from_json
from keras.models import load_model
import sounddevice as sd
import soundfile as sf
import os
import pickle
import tensorflow as tf
import librosa
import soundfile
import os, glob, pickle
import matplotlib.pyplot as plt
from yellowbrick.regressor import prediction_error
import statsmodels.api as sm
from keras.utils.vis_utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import linear_model, datasets, metrics
from sklearn.neural_network import MLPClassifier
from sklearn import gaussian_process
from sklearn import datasets,linear_model
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import learning_curve
import scikitplot as skplt
from sklearn.svm import LinearSVC

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
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
train_audio_path='input/train/'
test_audio_path='input/wav_tests'
classes={
  '1':'anger',
  '2':'bordom',
  '3':'fear',
  '4':'happines',
  '5':'sadness',
  '6':'neutral'
}
# Load the Model back from file
# with open("model81pkl.pkl", 'rb') as file:
#     Pickled_LR_Model= pickle.load(file)


# print(Pickled_LR_Model)
# Python program to illustrate
# Append vs write mode
import time
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
def load_data2(test_size=1):
    x,y=[],[]
    waves = [f for f in os.listdir(test_audio_path+'/' ) if f.endswith('.wav')]
    for wav in waves:
        file =test_audio_path + '/' + wav
        feature = extract_feature(file, mfcc=True, chroma=False, mel=False)
        print(wav)
        x.append(feature)
        y.append(wav)
    return x,y

loaded_model = joblib.load('model.h5')
x,y=load_data2(test_size=1)
y_pred=predict(x)
CreateFile(y_pred)



