import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

quandl.ApiConfig.api_key = "cYe6mNAb8dtBp1N1mQc_"

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = ((df["Adj. High"]-df["Adj. Close"])/df["Adj. Close"])*100
df['PCT_CHANGE'] = ((df["Adj. Close"]-df["Adj. Open"])/df["Adj. Open"])*100

df = df[['Adj. Close','HL_PCT','PCT_CHANGE','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-999999, inplace=True)


#Predict data 1% of the length of the dataframe in advance.
forecast_out = int(math.ceil(0.01*len(df)))
#print("Days in advance: "+ str(forecast_out))

df['label'] = df[forecast_col].shift(-forecast_out)



#Features column, drop the label in our data set.
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X) #Increases processing time. Data has zero mean and unit variance.
#X_lately takes the last forecast_out days to the end of the X list.
X_lately = X[-forecast_out:]
#X becomes everything except for the last forecast_out days in the X array.
X = X[:-forecast_out]


df.dropna(inplace=True)
y = np.array(df['label'])

#print("x is this long: "+ str(len(x)))
#print("Y is this long: " + str(len(y)))

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)

#Pickle the trained classifier so that retraining doesn't need
#to happen all of the time.
with open("linearregression.pickle", "wb") as f:
    pickle.dump(clf, f)

pickle_in = open("linearregression.pickle", "rb")
clf = pickle.load(pickle_in)   
    
accuracy = clf.score(X_test, y_test)

#Predicting the next forecast_out days into the future.
forecast_set = clf.predict(X_lately)
#print(forecast_set, accuracy, forecast_out)

df["Forecast"] = np.nan

#Indexing of the dataframe to the last date in the dataframe.
last_date = df.iloc[-1].name
#Changes the date into a datetime timestamp UTC which is in seconds.
last_unix = last_date.timestamp()
one_day = 86400 #seconds
next_unix = last_unix + one_day

#Loop through all predictions in the forecast_set, would be forecast_out days long.
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    #Access data frame by date. Assigns np.nan to all columns that are not
    #the predicted forecast. 
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df["Adj. Close"].plot()
df["Forecast"].plot()
plt.legend(loc = 4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
