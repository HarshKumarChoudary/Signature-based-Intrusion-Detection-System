from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd

class scaler(object):
    def __init__(self):
        data = pd.read_csv("feature.csv")
        data.replace("Benign_list_big_final","Benign",inplace=True)
        data.replace("Malware_dataset","Malware",inplace=True)
        data.replace("phishing_dataset","Phishing",inplace=True)
        data.replace("spam_dataset","Spam",inplace=True)
        data.replace(True,1,inplace = True)    
        data.replace(False,0,inplace = True)
        data.drop(columns='Unnamed: 0',inplace=True)
        data = data.drop(columns = "File")
        
        scaler = MinMaxScaler(feature_range=(0, 1))    
        self.scale_fit = scaler.fit(data)

    def scale(self,data):
        #save the mean and std. dev computed for your data.
        scaled_data = self.scale_fit.transform(data)
        return scaled_data
    