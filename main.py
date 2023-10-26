import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif

# Шаг 1: Загрузка и предварительный анализ данных
data = pd.read_csv("housing.csv")

# Шаг 2: Подготовка данных
data['price_category'] = (data['median_house_value'] >= 350000).astype(int)
data['near_ocean'] = ((data['ocean_proximity'] == 'NEAR OCEAN') | (data['ocean_proximity'] == 'NEAR BAY')).astype(int)

# Определение названий классов
class_labels = {0: "Низкая Цена", 1: "Высокая Цена"}

# Оценка информативности признаков
X = data[['price_category', 'near_ocean']]
y = data['price_category']
informative_scores = mutual_info_classif(X, y, discrete_features=[1])

# Вывод информативности признаков
for i, score in enumerate(informative_scores):
    print(f"Информативность признака {X.columns[i]}: {score:.4f}")

# Шаг 3: Разделение данных на обучающий и тестовый наборы
X = data[['price_category', 'near_ocean']]
y = data['price_category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 4: Построение модели дерева решений
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Шаг 6: Оценка показателей качества модели
y_pred = model.predict(X_test)
confusion = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Confusion Matrix:")
print(confusion)
print("Accuracy:", accuracy)

# Шаг 8: Визуализация решения
plt.figure(figsize=(10, 8))

# Разделим точки на два класса
high_price_near_ocean = data[(data['price_category'] == 1) & (data['near_ocean'] == 1)]
low_price_near_ocean = data[(data['price_category'] == 0) & (data['near_ocean'] == 1)]
high_price_not_near_ocean = data[(data['price_category'] == 1) & (data['near_ocean'] == 0)]
low_price_not_near_ocean = data[(data['price_category'] == 0) & (data['near_ocean'] == 0)]

plt.scatter(high_price_near_ocean['longitude'], high_price_near_ocean['latitude'], c='r', marker='o', label='Высокая Цена, Близко к Морю')
plt.scatter(low_price_near_ocean['longitude'], low_price_near_ocean['latitude'], c='b', marker='x', label='Низкая Цена, Близко к Морю')
plt.scatter(high_price_not_near_ocean['longitude'], high_price_not_near_ocean['latitude'], c='g', marker='s', label='Высокая Цена, Не Близко к Морю')
plt.scatter(low_price_not_near_ocean['longitude'], low_price_not_near_ocean['latitude'], c='y', marker='d', label='Низкая Цена, Не Близко к Морю')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'Decision Tree Classifier (Accuracy: {accuracy:.2f})')

plt.legend()
plt.show()
