import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Загружаем данные
bank_data = pd.read_csv('bank.csv', sep = ';')

# Отделяем значения признаков от результата (y - результат)
bank_features = bank_data.drop('y', axis = 1)
bank_output = bank_data.y

# Многие признаки у нас имеют строковые значения и их для регрессии необходимо преобразовать в числа
bank_features = pd.get_dummies(bank_features)
# Результат также переводим в число
bank_output = bank_output.replace({'no': 0, 'yes': 1})

# Разбиваем дата-сет на части - 75% для обучения, 25% - для проверки
X_train, X_test, y_train, y_test = train_test_split(bank_features, bank_output, test_size=0.25, random_state=42)

# Создаем модель
bank_model = LogisticRegression(C = 1e6, solver='liblinear')
bank_model.fit(X_train, y_train)

# Рассчитываем полученную точность
accuracy_score = bank_model.score(X_train, y_train)
print(accuracy_score)

# Демонстрация проблем с данными - данные не равномерные, что приводит к невысокойточности
plt.bar([0, 1], [len(bank_output[bank_output == 0]), len(bank_output[bank_output == 1])])
plt.xticks([0, 1])
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

print('Positive cases: {:.3f}% of all'.format(bank_output.sum() / len(bank_output) * 100))

# На тестовой части проводим прогнозирование
predictions = bank_model.predict(X_test)

# Сверяем прогнозы с данными и выводим отчет
# from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))