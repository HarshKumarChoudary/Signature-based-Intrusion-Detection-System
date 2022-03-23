from operator import imod
import tqdm
import pandas as pd
from tqdm import tqdm
from interruptingcow import timeout
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import pickle
import tensorflow as tf

data = pd.read_csv("feature.csv")
data.replace("Benign_list_big_final","Benign",inplace=True)
data.replace("Malware_dataset","Malware",inplace=True)
data.replace("phishing_dataset","Phishing",inplace=True)
data.replace("spam_dataset","Spam",inplace=True)
data.replace(True,1,inplace = True)    
data.replace(False,0,inplace = True)
data.drop(columns='Unnamed: 0',inplace=True)

y = data["File"]    

data = data.drop(columns = "File")

encoder = LabelEncoder()    
encoder.fit(y)    
Y = encoder.transform(y)  
   
scaler = MinMaxScaler(feature_range=(0, 1))    
X = scaler.fit_transform(data)
X = pd.DataFrame(X)

input_dim = len(data.columns)

model = Sequential()
model.add(Dense(256, input_dim = input_dim , activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(5, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

model.fit(X_train,np_utils.to_categorical(y_train),epochs = 50,validation_split=0.3, batch_size = 128)

# pickle.dump(model, open('model.pkl', 'wb'))
model_dir = "./mnist_model"

localhost_save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
model.save(model_dir, options=localhost_save_option)