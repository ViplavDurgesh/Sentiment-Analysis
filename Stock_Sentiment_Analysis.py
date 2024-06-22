import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer

# Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC 

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve, roc_auc_score
from sklearn import metrics
import itertools

import matplotlib.pyplot as plt

import tensorflow
from tensorflow import keras

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

df = pd.read_csv('E:\Finance_stock_sentiment\Data.csv', encoding = "ISO-8859-1")
df.head(1)   
train = df[df['Date'] < '20140101']
test = df[df['Date'] > '20121029']

# Removing punctuations
data = train.iloc[:, 2:27]
data.replace("[^a-zA-Z]", " ", regex=True, inplace=True) 

# Renaming column names for ease of access
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
data.head(5)

# Convertng headlines to lower case
for index in new_Index:
    data[index]=data[index].str.lower()
data.head(1)
headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))

train['headlines'] = headlines
train_headlines = headlines
selected_categories = ['headlines']

# Implement BAG OF WORDS
countvector = CountVectorizer(ngram_range=(2,2))
train_dataset_cv = countvector.fit_transform(headlines)

# implement RandomForest Classifier
rf=RandomForestClassifier(n_estimators=200,criterion='entropy')
rf.fit(train_dataset_cv,train['Label'])

# implement Logestic Regression
lr=LogisticRegression()
lr.fit(train_dataset_cv,train['Label'])

# implement K-Nearest Neighbor
knn=KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )
knn.fit(train_dataset_cv,train['Label'])

# implement Support vector classifier
svm = SVC(kernel='linear', random_state=0)  
svm.fit(train_dataset_cv,train['Label'])

# Transform Test dataset
test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_headlines = test_transform
test_dataset_cv = countvector.transform(test_transform)

# Prediciton RandomForest Classifier
predictions_rf_cv = rf.predict(test_dataset_cv)
matrix_rf_cv=confusion_matrix(test['Label'],predictions_rf_cv)
print("Confusion Mtrix :",matrix_rf_cv)
score_rf_cv=accuracy_score(test['Label'],predictions_rf_cv)
print("Accuracy Score : ",score_rf_cv,'\n')
report_rf_cv=classification_report(test['Label'],predictions_rf_cv)
print(report_rf_cv)

# Calculate ROC and AUC
probabilities_rf_cv = rf.predict_proba(test_dataset_cv)[:, 1]  # Get the probabilities of the positive class
fpr_rf_cv, tpr_rf_cv, _ = roc_curve(test['Label'], probabilities_rf_cv)
roc_auc_rf_cv = roc_auc_score(test['Label'], probabilities_rf_cv)

print("ROC AUC Score:", roc_auc_rf_cv)

# Plot ROC curve
plt.figure()
plt.plot(fpr_rf_cv, tpr_rf_cv, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_rf_cv)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Prediciton Logistic Regression
predictions_lr_cv = lr.predict(test_dataset_cv)
# CV-LR
matrix_lr_cv=confusion_matrix(test['Label'],predictions_lr_cv)
print("Confusion Matrix :")
print(matrix_lr_cv,'\n')
score_lr_cv=accuracy_score(test['Label'],predictions_lr_cv)
print("Accuracy Score:",score_lr_cv,'\n')
report_lr_cv=classification_report(test['Label'],predictions_lr_cv)
print(report_lr_cv)
# Calculate ROC and AUC
probabilities_lr_cv = lr.predict_proba(test_dataset_cv)[:, 1]  # Get the probabilities of the positive class
fpr_lr_cv, tpr_lr_cv, _ = roc_curve(test['Label'], probabilities_lr_cv)
roc_auc_lr_cv = roc_auc_score(test['Label'], probabilities_lr_cv)

print("ROC AUC Score:", roc_auc_lr_cv)

# Plot ROC curve
plt.figure()
plt.plot(fpr_lr_cv, tpr_lr_cv, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_lr_cv)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Prediciton K-Nearest Neighbor
predictions_knn_cv = knn.predict(test_dataset_cv)
# CV-KNN
matrix_knn_cv=confusion_matrix(test['Label'],predictions_knn_cv)
print("Confusion Matrix: ")
print(matrix_knn_cv,'\n')
score_knn_cv=accuracy_score(test['Label'],predictions_knn_cv)
print("Accuracy Score:",score_knn_cv,'\n')
report_knn_cv=classification_report(test['Label'],predictions_knn_cv)
print(report_knn_cv)

# Calculate ROC and AUC
probabilities_knn_cv = knn.predict_proba(test_dataset_cv)[:, 1]  # Get the probabilities of the positive class
fpr_knn_cv, tpr_knn_cv, _ = roc_curve(test['Label'], probabilities_knn_cv)
roc_auc_knn_cv = roc_auc_score(test['Label'], probabilities_knn_cv)

print("ROC AUC Score:", roc_auc_knn_cv)

# Plot ROC curve
plt.figure()
plt.plot(fpr_knn_cv, tpr_knn_cv, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_knn_cv)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Prediciton Support vector classifier
predictions_svm_cv = svm.predict(test_dataset_cv)
# CV-SVM
matrix_svm_cv=confusion_matrix(test['Label'],predictions_svm_cv)
print("Confusion Matrix:")
print(matrix_svm_cv,'\n')
score_svm_cv=accuracy_score(test['Label'],predictions_svm_cv)
print("Accuracy Score: ",score_svm_cv,'\n')
report_svm_cv=classification_report(test['Label'],predictions_svm_cv)
print(report_svm_cv)
# Calculate ROC and AUC
probabilities_svm_cv = rf.predict_proba(test_dataset_cv)[:, 1]  # Get the probabilities of the positive class
fpr_svm_cv, tpr_svm_cv, _ = roc_curve(test['Label'], probabilities_svm_cv)
roc_auc_svm_cv = roc_auc_score(test['Label'], probabilities_svm_cv)

print("ROC AUC Score:", roc_auc_svm_cv)
# Plot ROC curve
plt.figure()
plt.plot(fpr_svm_cv, tpr_svm_cv, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_svm_cv)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

import yfinance as yf

# Download the VADER lexicon
nltk.download('vader_lexicon')



# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Create a sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Compute sentiment scores
df['sentiment'] = df.apply(lambda row: sid.polarity_scores(' '.join(str(row[col]) for col in df.columns[2:]))['compound'], axis=1)

# Define sentiment labels
df['Label'] = df['sentiment'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

# Load stock data
stock_data = yf.download("AAPL", start="2000-01-01", end="2012-12-12")

# Reset index of stock_data
stock_data = stock_data.reset_index()

# Merge stock data with sentiment data on 'Date'
stock_data = pd.merge(stock_data, df[['Date', 'Label']], on='Date', how='left')

# Fill missing sentiment scores with 0 (or another strategy if preferred)
stock_data['Label'] = stock_data['Label'].fillna(0)

# Identify buy/sell points based on sentiment labels
buy_signals = stock_data[stock_data['Label'] > 0.5]
sell_signals = stock_data[stock_data['Label'] < -0.5]

# Plot buy/sell signals on stock price chart
plt.figure(figsize=(14, 7))
plt.plot(stock_data['Close'], label='Stock Price')
plt.scatter(buy_signals.index, stock_data.loc[buy_signals.index]['Close'], marker='^', color='g', label='Buy Signal', alpha=1)
plt.scatter(sell_signals.index, stock_data.loc[sell_signals.index]['Close'], marker='v', color='r', label='Sell Signal', alpha=1)
plt.title('Stock Price with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Calculate portfolio returns and other performance metrics
initial_balance = 100000  # Example initial balance
balance = initial_balance
positions = 0

for date in stock_data.index:
    if date in buy_signals.index:
        positions += balance // stock_data.loc[date]['Close']
        balance %= stock_data.loc[date]['Close']
    elif date in sell_signals.index and positions > 0:
        balance += positions * stock_data.loc[date]['Close']
        positions = 0

final_balance = balance + positions * stock_data.iloc[-1]['Close']
returns = (final_balance - initial_balance) / initial_balance

print("Final Portfolio Value: ${:.2f}".format(final_balance))
print("Returns: {:.2f}%".format(returns * 100))


# Additional performance metrics
# Sharpe Ratio
daily_returns = stock_data['Close'].pct_change().dropna()
excess_returns = daily_returns - 0.01 / 252  # Assuming 1% risk-free rate annually
sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
print("Sharpe Ratio:", sharpe_ratio)

# Maximum Drawdown
rolling_max = stock_data['Close'].cummax()
daily_drawdown = stock_data['Close'] / rolling_max - 1.0
max_drawdown = daily_drawdown.cummin().min()
print("Maximum Drawdown:", max_drawdown)


