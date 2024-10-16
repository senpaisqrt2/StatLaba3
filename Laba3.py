import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

# Шаг 1: Загрузка данных из файла "Москва_2021.txt"
# Открываем файл, где содержатся возраста преступников, совершивших преступления в Москве в 2021 году.
# Каждый возраст повторяется столько раз, сколько преступлений совершили люди этого возраста.
file_path = "C:\\Users\\nexti\\Desktop\\Uni\\3 course\\Stat\\Laba2\\Москва_2021.txt"
with open(file_path, 'r') as file:
    # Читаем файл построчно, удаляем лишние пробелы (strip) и преобразуем строки в числа (int)
    crime_ages = [int(line.strip()) for line in file]

# Преобразуем список возрастов в DataFrame для удобства анализа
ages_df = pd.DataFrame(crime_ages, columns=["Age"])

# Выводим несколько статистик по данным для общей информации
print("Всего данных (преступников):", len(ages_df))  # Сколько всего записей в наборе данных
print("Минимальный возраст:", ages_df["Age"].min())  # Минимальный возраст преступника
print("Максимальный возраст:", ages_df["Age"].max())  # Максимальный возраст преступника
print("Средний возраст:", ages_df["Age"].mean())  # Средний возраст (математическое ожидание)
print("Стандартное отклонение возраста:", ages_df["Age"].std())  # Стандартное отклонение

# Параметры для расчетов (из лабораторной работы)
gamma = 0.95  # Доверительная вероятность
delta = 3  # Точность оценки математического ожидания (в годах)
z_value = stats.norm.ppf((1 + gamma) / 2)  # Квантиль нормального распределения для доверительной вероятности
sigma = np.std(crime_ages, ddof=0)  # Стандартное отклонение генеральной совокупности

# Функция для расчета объема выборки
def calculate_sample_size(sigma, delta, gamma, z_value):
    return (z_value * sigma / delta) ** 2

# Расчет объема выборки
n = int(np.ceil(calculate_sample_size(sigma, delta, gamma, z_value)))
print(f"Рассчитанный объем выборки: {n}")

# Генерация 36 выборок и расчет выборочных средних (как в лабораторной №2)
sample_means = [np.mean(np.random.choice(crime_ages, size=n, replace=True)) for _ in range(36)]

# Шаг 2: Проверка гипотезы о нормальном распределении выборочных средних (Задача 1б)
# Разбиваем выборочные средние на 7 равных интервалов (групп)
intervals = pd.cut(sample_means, bins=7, include_lowest=True)

# Считаем наблюдаемые частоты
observed_frequencies = intervals.value_counts().sort_index()

# Среднее и стандартное отклонение выборочных средних
mean_sample = np.mean(sample_means)
std_sample = np.std(sample_means, ddof=1)

# Расчет ожидаемых частот для нормального распределения
expected_frequencies = [stats.norm.cdf((max_interval.right - mean_sample) / std_sample) -
                        stats.norm.cdf((max_interval.left - mean_sample) / std_sample)
                        for max_interval in observed_frequencies.index]

# Преобразуем вероятности в абсолютные значения, умножив на общее количество средних
expected_frequencies = np.array(expected_frequencies) * len(sample_means)

# Нормализуем ожидаемые частоты, чтобы их сумма совпадала с суммой наблюдаемых частот
expected_frequencies = expected_frequencies * (observed_frequencies.sum() / expected_frequencies.sum())

# Применяем критерий Пирсона
chi2_stat, p_value = stats.chisquare(f_obs=observed_frequencies, f_exp=expected_frequencies)

# Вывод результатов
print("Критерий Пирсона для выборочных средних:")
print(f"Chi2 статистика: {chi2_stat}")
print(f"p-value: {p_value}")

# Принятие или отклонение гипотезы
alpha = 0.05
if p_value < alpha:
    print("Нулевая гипотеза о нормальном распределении отклоняется.")
else:
    print("Нет оснований отклонять нулевую гипотезу о нормальном распределении.")

# Шаг 3: Проверка гипотезы о равенстве дисперсий двух выборок (Задача 2)
# Генерируем две случайные выборки, как это было сделано в лабораторной работе №2
sample1 = np.random.choice(crime_ages, size=n, replace=True)
sample2 = np.random.choice(crime_ages, size=n, replace=True)

# Вычисление дисперсий для каждой выборки
var1 = np.var(sample1, ddof=1)
var2 = np.var(sample2, ddof=1)

# Применение критерия Фишера для проверки гипотезы о равенстве дисперсий
f_stat = var1 / var2 if var1 > var2 else var2 / var1  # Рассчитываем F-статистику (отношение дисперсий)
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

# Дополнительные выводы для наглядности:
print("Среднее значение возраста:", mean_sample)  # Средний возраст
print("Стандартное отклонение выборочных средних:", std_sample)  # Стандартное отклонение выборочных средних
print("Сумма наблюдаемых частот:", observed_frequencies.sum())  # Сумма наблюдаемых частот
print("Сумма ожидаемых частот:", expected_frequencies.sum())  # Сумма ожидаемых частот
print("Проверка равенства сумм наблюдаемых и ожидаемых частот:", np.isclose(observed_frequencies.sum(), expected_frequencies.sum()))
