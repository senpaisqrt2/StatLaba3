import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
file_path = "C:\\Users\\nexti\\Desktop\\Uni\\3 course\\Stat\\Laba2\\Москва_2021.txt"
with open(file_path, 'r') as file:
    crime_ages = [int(line.strip()) for line in file]

ages_df = pd.DataFrame(crime_ages, columns=["Age"])

# Выводим несколько статистик по данным для общей информации
print("Всего данных (преступников):", len(ages_df))  # Сколько всего записей в наборе данных
print("Минимальный возраст:", ages_df["Age"].min())  # Минимальный возраст преступника
print("Максимальный возраст:", ages_df["Age"].max())  # Максимальный возраст преступника
print("Стандартное отклонение возраста:", ages_df["Age"].std())  # Стандартное отклонение

# Параметры задачи
gamma = 0.95
delta = 3
z_value = stats.norm.ppf((1 + gamma) / 2)
sigma = np.std(crime_ages, ddof=0)

alpha = 0.05

# Расчет объема выборки
def calculate_sample_size(sigma, delta, gamma, z_value):
    return (z_value * sigma / delta) ** 2

n = int(np.ceil(calculate_sample_size(sigma, delta, gamma, z_value)))
sample_means = [np.mean(np.random.choice(crime_ages, size=n, replace=True)) for _ in range(36)]

# --- Интервальный расчет среднего возраста ---
# Разбиваем возраста на интервалы для расчета среднего интервальным методом
bins = np.arange(0, ages_df["Age"].max() + 5, 5)  # Создаем интервалы по 5 лет
labels = [f'{int(bins[i])}-{int(bins[i+1])-1}' for i in range(len(bins)-1)]  # Метки для интервалов

# Разбиение возрастов на интервалы
ages_df['Age Group'] = pd.cut(ages_df['Age'], bins=bins, labels=labels, include_lowest=True)

# Подсчёт количества преступников в каждой возрастной группе
age_group_counts = ages_df['Age Group'].value_counts().sort_index()

# Рассчитываем середину каждого интервала (например, середина интервала 0-4 это 2.5)
mid_points = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]

# Рассчитываем средний возраст интервальным методом:
# Среднее значение = (частота * середина интервала) / общее количество наблюдений
total_count = age_group_counts.sum()
weighted_sum = sum(f * m for f, m in zip(age_group_counts, mid_points))

# Интервальное среднее
interval_mean = weighted_sum / total_count

# Вывод результата
print(f"Интервальное среднее возраста: {interval_mean:.2f}")

# --- Chi^2 для Задания 1а (проверка нормальности для исходных данных) ---
# Разбиваем исходные данные на 7 групп для критерия Пирсона
intervals_1a = pd.cut(ages_df['Age'], bins=7, include_lowest=True)
observed_frequencies_1a = intervals_1a.value_counts().sort_index()

# Среднее и стандартное отклонение для исходных данных
mean_age_1a = ages_df["Age"].mean()
std_age_1a = ages_df["Age"].std()

# Ожидаемые частоты для нормального распределения (по исходным данным)
expected_frequencies_1a = [stats.norm.cdf((interval.right - mean_age_1a) / std_age_1a) -
                           stats.norm.cdf((interval.left - mean_age_1a) / std_age_1a)
                           for interval in observed_frequencies_1a.index]
expected_frequencies_1a = np.array(expected_frequencies_1a) * len(ages_df)

# Нормализуем ожидаемые частоты
expected_frequencies_1a = expected_frequencies_1a * (observed_frequencies_1a.sum() / expected_frequencies_1a.sum())

# Шаг 1: Рассчитаем chi^2 для задания 1а вручную
chi2_manual_1a = np.sum((observed_frequencies_1a - expected_frequencies_1a) ** 2 / expected_frequencies_1a)
print(f"Вычисленная chi^2 (Задание 1а): {chi2_manual_1a}")

# Шаг 2: Найдем критическое значение chi^2 для задания 1а
df_1a = len(observed_frequencies_1a) - 1  # Степени свободы = число групп - 1
chi2_critical_1a = stats.chi2.ppf(1 - alpha, df_1a)

print(f"Критическое значение chi^2 для df={df_1a} и уровня значимости 0.05 (Задание 1а): {chi2_critical_1a}")

# Шаг 3: Сравним рассчитанную chi^2 со значением критической chi^2
if chi2_manual_1a > chi2_critical_1a:
    print("Нулевая гипотеза о нормальном распределении для исходных данных отклоняется (Задание 1а).")
else:
    print("Нет оснований отклонять нулевую гипотезу о нормальном распределении для исходных данных (Задание 1а).")

# --- Chi^2 для Задания 1б (выборочные средние) ---
# Разбиение выборочных средних на 7 групп
intervals_1b = pd.cut(sample_means, bins=7, include_lowest=True)
observed_frequencies_1b = intervals_1b.value_counts().sort_index()

# Среднее и стандартное отклонение выборочных средних
mean_sample = np.mean(sample_means)
std_sample = np.std(sample_means, ddof=1)

# Ожидаемые частоты для нормального распределения (по выборочным средним)
expected_frequencies_1b = [stats.norm.cdf((interval.right - mean_sample) / std_sample) -
                           stats.norm.cdf((interval.left - mean_sample) / std_sample)
                           for interval in observed_frequencies_1b.index]
expected_frequencies_1b = np.array(expected_frequencies_1b) * len(sample_means)

# Нормализация ожидаемых частот
expected_frequencies_1b = expected_frequencies_1b * (observed_frequencies_1b.sum() / expected_frequencies_1b.sum())

# Шаг 1: Рассчитаем chi^2 для задания 1б вручную
chi2_manual_1b = np.sum((observed_frequencies_1b - expected_frequencies_1b) ** 2 / expected_frequencies_1b)
print(f"Вычисленная chi^2 (Задание 1б): {chi2_manual_1b}")

# Шаг 2: Найдем критическое значение chi^2 для задания 1б
df_1b = len(observed_frequencies_1b) - 1  # Степени свободы = число групп - 1
chi2_critical_1b = stats.chi2.ppf(1 - alpha, df_1b)

print(f"Критическое значение chi^2 для df={df_1b} и уровня значимости 0.05 (Задание 1б): {chi2_critical_1b}")

# Шаг 3: Сравним рассчитанную chi^2 со значением критической chi^2
if chi2_manual_1b > chi2_critical_1b:
    print("Нулевая гипотеза о нормальном распределении для выборочных средних отклоняется (Задание 1б).")
else:
    print("Нет оснований отклонять нулевую гипотезу о нормальном распределении для выборочных средних (Задание 1б).")

# --- Проверка гипотезы о равенстве дисперсий для Задания 2 ---
# Генерируем две случайные выборки
sample1 = np.random.choice(crime_ages, size=n, replace=True)
sample2 = np.random.choice(crime_ages, size=n, replace=True)

# Вычисление дисперсий для каждой выборки
var1 = np.var(sample1, ddof=1)
var2 = np.var(sample2, ddof=1)

# Применение критерия Фишера для проверки гипотезы о равенстве дисперсий
f_stat = var1 / var2 if var1 > var2 else var2 / var1  # Рассчитываем F-статистику
df1 = len(sample1) - 1  # Степени свободы для первой выборки
df2 = len(sample2) - 1  # Степени свободы для второй выборки
p_value_f = 1 - stats.f.cdf(f_stat, df1, df2)  # Вычисляем p-value для F-критерия

# Вывод результатов критерия Фишера
print("Критерий Фишера для проверки равенства дисперсий:")
print(f"F-статистика: {f_stat}")
print(f"p-value: {p_value_f}")

# Принятие или отклонение гипотезы о равенстве дисперсий
if p_value_f < alpha:
    print("Нулевая гипотеза о равенстве дисперсий отклоняется.")
else:
    print("Нет оснований отклонять нулевую гипотезу о равенстве дисперсий.")

# Дополнительные выводы
print("Среднее значение возраста (по выборочным средним):", mean_sample)
print("Стандартное отклонение выборочных средних:", std_sample)
