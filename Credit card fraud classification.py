import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


PATH = 'D:/pjatk/PAD/PAD_ZALICZENIE/creditcard.csv'
df = pd.read_csv(PATH, sep=',')
print(df.head())

df.isnull().sum()
df.drop_duplicates(inplace=True)
print(df.info)
print(df.shape)
print(df.describe())

df.hist(figsize=(20,20))
plt.show()

print(df['Class'].value_counts())

count_classes = pd.value_counts(df['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

# Określenie ile jest nieuprawnonych transakcji w zbiorze

fraud = df[df['Class'] == 1]
valid = df[df['Class'] == 0]

outlierFraction = len(fraud)/float(len(valid))

print(outlierFraction)
print('Transakcja nieuprawniona : {}'.format(len(df[df['Class'] == 1])))
print('Transakcja : {}'.format(len(df[df['Class'] == 0])))

print('Ilosc nieuprawnionych transakcji:{}'
      .format(fraud.Amount.describe()))


# heatmap

corrmat = df.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()

#sample

df1 = df.sample(frac = 0.1, random_state = 1)
print(df1.shape)

# Podział na część treningową i testową

y = df1['Class'].values #target
X = df1.drop(['Class'],axis=1).values #features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# regresja logistyczna

model1 = LogisticRegression()
model1.fit(X_train, y_train)
predict = model1.predict(X_test)
print(model1.coef_)

# walidacja
print(metrics.accuracy_score(y_test, predict))
print(metrics.precision_score(y_test, predict))
print(metrics.recall_score(y_test, predict))
print(metrics.classification_report(y_test, predict))

roc = metrics.roc_curve(y_test, predict, pos_label=1)
fpr = roc[0]
tpr = roc[1]
thrs = roc[2]
plt.xlim(0.0,1.0)
plt.ylim(0.0,0.1)
plt.plot(fpr,tpr)
plt.plot([0.0,1.0],[0.0,1.0],linestyle='--')
plt.show()

#Klasyfikator Naiwny Bayes

model2 = GaussianNB()
model2.fit(X_train,y_train)
predict2 = model2.predict(X_test)


# walidacja
print(metrics.accuracy_score(y_test, predict))
print(metrics.precision_score(y_test, predict))
print(metrics.recall_score(y_test, predict))
print(metrics.classification_report(y_test, predict))

roc = metrics.roc_curve(y_test, predict, pos_label=1)
fpr = roc[0]
tpr = roc[1]
thrs = roc[2]
plt.xlim(0.0,1.0)
plt.ylim(0.0,0.1)
plt.plot(fpr,tpr)
plt.plot([0.0,1.0],[0.0,1.0],linestyle='--')
plt.show()
