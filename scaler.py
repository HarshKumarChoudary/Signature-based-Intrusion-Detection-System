from sklearn.preprocessing import LabelEncoder, MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))    
X = scaler.fit_transform(data)
X = pd.DataFrame(X)