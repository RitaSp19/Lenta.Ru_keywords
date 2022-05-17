# Lenta.Ru_keywords
Цель проекта - определение самых частотных тем новостей портала Lenta.ru для каждого месяца.

Для решения этой задачи я выбрала метод нахождения ключевых слов с помощью алгоритма tf-idf.

Предобработка текста:
- Удаление лишних символов - использовала реулярные выражения)
- Удаление стоп-слов (взяла список из корпусов nltk, так как это было самым быстрым способом)
- Лемматизация - использовала Mystem из-за большей точности, чем при использовании pymorphy2 (хотя Mystem работает медленнее).
Лемматизация занимает время, поэтому я выполнила задание не для всего корпуса, а только для первых 2000 строк.

Использовала TfidfVectorizer из библиотеки sklearn. Применила TfidfVectorizer ко всему корпусу новостей и получила список ключевых слов по всему корпусу.
Затем я распределила ключевые слова по месяцам и выделила самые частотные для каждого месяца.

Использованные библиотеки: pandas, pymystem3, re, sklearn, nltk

Статус проекта: нуждается в доработке.

Что можно попробовать для усовершенствования метода на этапе предобработки:

1) Можно добавить в стоп-слова гиперонимы вроде обозначения людей ("мужчина", "человек", "президент"), названий областей ("страна", "штат").
Впрочем, эти слова скорее всего попадали в ключевые, потому что алгоритм был применен всего к 2000 элементам.
На всем корпусе эти слова могут отсеяться при исполнении tf-idf)

2) Извлечь и представить в виде токенов (перед применением tf-idf):

- названия в кавычках (вроде "Движение вверх") - можно сделать с помощью регулярных выражений,
- имена (Ким Чен Ын) и даты (9 мая, 11 сентября)

3) Использовать биграммы, добавив в TfidfVectorizer параметр ngram_range.

4) Если не создавать биграммы, можно оставить в ключевых словах только существительные, предварительно сделав морфологичеcкий анализ (самая сомнительная идея)

5) Доработать функцию clean_text, добавить длинное тире и т.п.

6) Можно каким-то образом использовать заголовок: например, выделить из него частотные слова...
