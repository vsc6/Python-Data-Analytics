import string
import nltk
import pandas as pd
from nltk.corpus import stopwords

dane = pd.read_csv('emails.csv')
dane.head()
print(dane.shape)
print(dane.columns)
dane.drop_duplicates(inplace=True)
print(dane.shape)

# wykres
import plotly_express as px

fig = px.histogram(dane, x = 'spam', color='spam',
                  title='Zbiór danych')
fig.show()

# brakujące wartości
dane.isnull().sum()

# usuwanie niepotrzebnych słów
nltk.download('stopwords')
print(dane.head)

#tokenizacja
def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

 #pokaż listę tokenów

print(dane['text'].head().apply(process_text))


from sklearn.feature_extraction.text import CountVectorizer
messages_bow = CountVectorizer(analyzer=process_text).fit_transform(dane['text'])
messages_bow.shape

print(messages_bow.shape)

# zmienna celu

y = df1['Class'].values #target
X = df1.drop(['Class'],axis=1).values #features

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()

(classifier.fit(X_train, y_train))
print(classifier.predict(X_train))
print(y_train.values)

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_train)
print(classification_report(y_train ,pred ))
print('Confusion Matrix: \n',confusion_matrix(y_train,pred))
print()
print('Accuracy: ', accuracy_score(y_train,pred))

print('Predicted value: ',classifier.predict(X_test))
print('Actual value: ',y_test.values)

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_test)
print(classification_report(y_test ,pred ))

print('Confusion Matrix: \n', confusion_matrix(y_test,pred))
print()
print('Accuracy: ', accuracy_score(y_test,pred))