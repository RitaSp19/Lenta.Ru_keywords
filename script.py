import pandas as pd
from pymystem3 import Mystem
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from collections import Counter
from nltk.corpus import stopwords

dataset = pd.read_csv('lenta-ru-news_19-21_raw.csv')
dataset.head()

file_texts_all = dataset.text.values.tolist()
file_texts = file_texts_all[0: 2000] # Обрезала датасет (из-за долгой лемматизации)
#file_texts = dataset.text.values.tolist() # Если анализировать датасет полностью
file_texts

def clean_text(text):

    # оставила кавычки-ёлочки (в них находятся названия, которые могут пригодиться при улучшении скрипта)
    # оставила латинские символы, потому что могут оказаться важными аббревиатцры вроде UEFA
    
    text = text.lower() # Можно переделать на изменение заглавной буквы только в начале предложения (если потребуется выделить имена)
    regular = r'[\*+\#+\№\"\-+\+\=+\?+\&\^\.+\;\,+\>+\(\)\/+\:\\+]'
    text = re.sub(regular, ' ', text)
    text = re.sub(r'(\d+\s\d+)|(\d+)', ' ', text) # Но: пропадают потенциальные ключевые слова вроде 9 Мая, 8 Марта, 23 Февраля
    text = re.sub(r'\s+', ' ', text)
    
    return text

cleaned_text = []

for text in file_texts:
    text = clean_text(text)
    cleaned_text.append(text)
    
cleaned_text

rus_stops = set(stopwords.words('russian'))
rus_stops

analizator = Mystem()

def preprocess_for_tfidf(some_text):
    lemmatized_text = analizator.lemmatize(some_text)
    return ' '.join(lemmatized_text)

def produce_tf_idf_keywords(some_text, number_of_words):
    make_tf_idf = TfidfVectorizer(stop_words = rus_stops)
    text_as_tf_idf_vectors = make_tf_idf.fit_transform(preprocess_for_tfidf(text) for text in some_text)
    id2word = {i: word for i, word in enumerate(make_tf_idf.get_feature_names())}
    
    list_tf_idf = []
    
    for text_row in range(text_as_tf_idf_vectors.shape[0]):
        row_data = text_as_tf_idf_vectors.getrow(text_row)
        words_for_this_text = row_data.toarray().argsort()
        top_word_for_this_text = words_for_this_text[0, -(number_of_words): -1]
        list_tf_idf.append([id2word[w] for w in top_word_for_this_text])
    
    return list_tf_idf

tf_idf_keywords = produce_tf_idf_keywords(cleaned_text, 11)
tf_idf_keywords

url_all = dataset.url.values.tolist()
url = url_all[0: 2000] # Обрезала, так как д.б. равен len(file_texts)

# url = dataset.url.values.tolist() # Для всего датасета

url_with_regexp = []

for i in url:
    year = re.search(r'\d{4}', i)
    i_with_deleted_year = re.sub('\d{4}', r'MOUNTH', i)
    month = re.search(r'\d{2}', i_with_deleted_year)
    url_with_regexp.append(month[0] + '.' + year[0]

month_and_year = url_with_regexp

month_keywords = defaultdict(list)

for (m, kws) in zip(month_and_year, tf_idf_keywords):
    month_keywords[m].extend(kws)

print(month_keywords)

column1 = list(month_keywords.keys())
column2 = list(month_keywords.values())

def most_frequent(words_to_sort):
    occurence_count = Counter(words_to_sort)
    return occurence_count.most_common(15)
column2_most_frequent = []

for i in range(len(column2)):

    words_to_sort = column2[i]
    current_column_most_frequent_tuple = most_frequent(words_to_sort)
    
    current_column_most_frequent_string = ''
    
    for j in range(len(current_column_most_frequent_tuple)):
        if j != len(current_column_most_frequent_tuple) - 1:
            a = list(current_column_most_frequent_tuple[j])
            current_column_most_frequent_string += (a[0] + ', ')
        else:
            a = list(current_column_most_frequent_tuple[j])
            current_column_most_frequent_string += (a[0])
    
    column2_most_frequent.append(current_column_most_frequent_string)

column2_most_frequent

result = pd.DataFrame({'month': column1, 'hot_topic': column2_most_frequent})
result.head()

result.to_csv(r'PATH', index=False) # заменить PATH на расположение файла