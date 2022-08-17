# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Dense
PredictorScaler=StandardScaler()
TargetVarScaler=StandardScaler()



 
df = pd.read_csv('train_data_evaluation_part_2.csv')
dataset = pd.read_csv('test_data_evaluation_part2.csv')

df.drop(columns='Unnamed: 0',inplace=True)
dataset.drop(columns='Unnamed: 0',inplace=True)

df['Age'].fillna(df['Age'].mean(),inplace=True)
df['Age']=df['Age'].astype(int)

dataset['Age'].fillna(dataset['Age'].mean(),inplace=True)
dataset['Age']=dataset['Age'].astype(int)

columns=['Nationality','MarketSegment','DistributionChannel']
for feature in columns:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    dataset[feature] = le.fit_transform(dataset[feature])

X=df.drop(columns=['BookingsCheckedIn','ID'])
y=df[['BookingsCheckedIn']]

test_X=dataset.drop(columns=['BookingsCheckedIn','ID'])
test_y=dataset[['BookingsCheckedIn']]

## train data
PredictorScalerFit=PredictorScaler.fit(X)
TargetVarScalerFit=TargetVarScaler.fit(y)

X=PredictorScalerFit.transform(X)
y=TargetVarScalerFit.transform(y)
 

 
# create ANN model
model = Sequential()
model.add(Dense(units=32, input_dim=27, kernel_initializer='normal', activation='relu')) 
model.add(Dense(units=16, kernel_initializer='normal', activation='tanh'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')



epochs=10
batch_size=5
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=15, restore_best_weights=True)

history = model.fit(X,y,callbacks=[es],epochs=epochs,batch_size=batch_size,shuffle=True,validation_split=0.2,verbose=1)

## Model dump
pickle.dump(model, open('model.pkl','wb'))


## test data
PredictorScalerFit=PredictorScaler.fit(test_X)
TargetVarScalerFit=TargetVarScaler.fit(test_y)

## Loading model to compare the results
# model = pickle.load(open('model.pkl','rb'))

Predictions=model.predict(test_X)
Predictions=TargetVarScalerFit.inverse_transform(Predictions)
y_test_orig=TargetVarScalerFit.inverse_transform(test_y)
Test_Data=PredictorScalerFit.inverse_transform(test_X)
Predictors=['Nationality', 'Age', 'DaysSinceCreation', 'AverageLeadTime',
       'LodgingRevenue', 'OtherRevenue', 'BookingsCanceled',
       'BookingsNoShowed', 'PersonsNights', 'RoomNights',
       'DaysSinceLastStay', 'DaysSinceFirstStay', 'DistributionChannel',
       'MarketSegment', 'SRHighFloor', 'SRLowFloor', 'SRAccessibleRoom',
       'SRMediumFloor', 'SRBathtub', 'SRShower', 'SRCrib', 'SRKingSizeBed',
       'SRTwinBed', 'SRNearElevator', 'SRAwayFromElevator',
       'SRNoAlcoholInMiniBar', 'SRQuietRoom']
TestingData=pd.DataFrame(data=Test_Data, columns=Predictors)
TestingData['BookingsCheckedIn_y']=y_test_orig
TestingData['BookingsCheckedIn']=abs(np.round_(Predictions))

TestingData.fillna(0,inplace=True)
APE=100*(abs(TestingData['BookingsCheckedIn_y']-TestingData['BookingsCheckedIn'])/TestingData['BookingsCheckedIn_y'])
APE.replace([np.inf, -np.inf], 0, inplace=True)
TestingData['APE']=APE

lst=[]
accuracy =(100-abs(np.mean(APE)))

lst.append(accuracy)



dataset['BookingsCheckedIn_predicted']=abs(np.round_(Predictions))
dataset.drop(columns='BookingsCheckedIn').to_csv('Final_test_csv_with_predictions.csv',index=False)

dict = {'accuracy': lst}     
accu = pd.DataFrame(dict)
accu.to_csv('Accuracy.csv',index=False)

