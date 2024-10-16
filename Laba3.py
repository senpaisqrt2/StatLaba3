import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
file_path = "C:\\Users\\nexti\\Desktop\\Uni\\3 course\\Stat\\Laba2\\Москва_2021.txt"
with open(file_path, 'r') as file:
    crime_ages = [int(line.strip()) for line in file]

ages_df = pd.DataFrame(crime_ages, columns=["Age"])

# Параметры задачи
gamma = 0.95
delta = 3
alpha = 0.05  # Уровень значимости

# Стандартное отклонение выборки
sigma = np.std(crime_ages, ddof=0)

# Расчет объема выборки
z_value = stats.norm.ppf((1 + gamma) / 2)


def calculate_sample_size(sigma, delta, gamma, z_value):
    return (z_value * sigma / delta) ** 2


n = int(np.ceil(calculate_sample_size(sigma, delta, gamma, z_value)))

# Генерация выборочных средних для задачи 1б
sample_means = [np.mean(np.random.choice(crime_ages, size=n, replace=True)) for _ in range(36)]



# --- Интервальный расчет среднего возраста ---
# Середины интервалов
mid_points = [18.5, 27.5, 36.5, 45.5, 54.5, 63.5, 72.5]

# Частоты для каждого интервала (как указано тобой)
age_group_counts = [4811, 9476, 7243, 7951, 1140, 1472, 330]

# Шаг 1: Умножаем середины интервалов на соответствующие частоты
weighted_sum = sum(f * m for f, m in zip(age_group_counts, mid_points))

# Шаг 2: Считаем общую сумму частот
total_count = sum(age_group_counts)

# Шаг 3: Делим взвешенную сумму на общую сумму частот
interval_mean = weighted_sum / total_count

# Вывод результата
print(f"Взвешенная сумма: {weighted_sum}")
print(f"Общая сумма частот: {total_count}")
print(f"Интервальное среднее возраста: {interval_mean:.2f}")
print(f" ")
print(f" ")
print(f" ")


print(f"1A")
# --- Chi^2 для Задания 1а (проверка нормальности для исходных данных) ---

# Наблюдаемые частоты (из данных)
observed_frequencies = np.array([4811, 9476, 7243, 7951, 1140, 1472, 330])

# Середины интервалов
mid_points = np.array([18.5, 27.5, 36.5, 45.5, 54.5, 63.5, 72.5])

# Среднее и стандартное отклонение реальных данных (рассчитаем их вместо фиксированных значений)
mean_age = 35.63  # Это среднее, рассчитанное вручную (по вашим данным)
std_age = np.std([18.5] * 4811 + [27.5] * 9476 + [36.5] * 7243 + [45.5] * 7951 +
                 [54.5] * 1140 + [63.5] * 1472 + [72.5] * 330, ddof=1)  # Стандартное отклонение

print(f"Рассчитанное стандартное отклонение: {std_age}")

# Общее количество наблюдений
total_count = observed_frequencies.sum()

# Расчет ожидаемых частот на основе нормального распределения
expected_frequencies = []
for i in range(len(mid_points)):
    # Определим границы интервалов
    if i == 0:
        lower_bound = -np.inf  # Для первого интервала нижняя граница бесконечность
    else:
        lower_bound = (mid_points[i - 1] + mid_points[i]) / 2

    if i == len(mid_points) - 1:
        upper_bound = np.inf  # Для последнего интервала верхняя граница бесконечность
    else:
        upper_bound = (mid_points[i] + mid_points[i + 1]) / 2

    # Используем нормальное распределение для расчета вероятности попадания в интервал
    prob = stats.norm.cdf(upper_bound, loc=mean_age, scale=std_age) - stats.norm.cdf(lower_bound, loc=mean_age,
                                                                                     scale=std_age)

    # Рассчитываем ожидаемую частоту для интервала
    expected_frequencies.append(prob * total_count)

expected_frequencies = np.array(expected_frequencies)

# Рассчитываем chi^2
chi2_manual_1a = np.sum((observed_frequencies - expected_frequencies) ** 2 / expected_frequencies)

# Вывод результатов
print(f"Наблюдаемые частоты: {observed_frequencies}")
print(f"Ожидаемые частоты: {expected_frequencies}")
print(f"Вычисленная chi^2: {chi2_manual_1a:.3f}")

# Количество степеней свободы для задания 1а = число интервалов - 1
df_1a = len(observed_frequencies) - 3  # Степени свободы = число групп - 1

# Критическое значение chi^2 для уровня значимости alpha
chi2_critical_1a = stats.chi2.ppf(1 - alpha, df_1a)

# Вывод критического значения chi^2
print(f"Критическое значение chi^2 для df={df_1a} и уровня значимости 0.05: {chi2_critical_1a:.3f}")

# Сравнение chi^2 с критическим значением
if chi2_manual_1a > chi2_critical_1a:
    print("Нулевая гипотеза о нормальном распределении для исходных данных отклоняется (Задание 1а).")
else:
    print("Нет оснований отклонять нулевую гипотезу о нормальном распределении для исходных данных (Задание 1а).")

print(f" ")
print(f" ")
print(f"1Б")
# Параметры выборки для выборочных средних
gamma = 0.95
delta = 3
alpha = 0.05

# Генерация выборочных средних для задачи 1б
n = 36  # количество выборок
sample_means = [np.mean(np.random.choice(crime_ages, size=n, replace=True)) for _ in range(36)]

# --- Chi^2 для Задания 1б (проверка нормальности для выборочных средних) ---
# Разбиваем выборочные средние на интервалы
intervals_1b = pd.cut(sample_means, bins=7, include_lowest=True)
observed_frequencies_1b = intervals_1b.value_counts().sort_index()

# Среднее и стандартное отклонение выборочных средних
mean_sample = np.mean(sample_means)  # Среднее выборочных средних
std_sample = np.std(sample_means, ddof=1)  # Стандартное отклонение выборочных средних

print(f"Среднее выборочных средних: {mean_sample}")
print(f"Стандартное отклонение выборочных средних: {std_sample}")

# Ожидаемые частоты для нормального распределения на основе выборочных средних
expected_frequencies_1b = []

# Найдем середины интервалов, используя атрибуты left и right из объекта intervals_1b.categories
mid_points_1b = [(interval.left + interval.right) / 2 for interval in intervals_1b.categories]

total_count_1b = observed_frequencies_1b.sum()

for i in range(len(mid_points_1b)):
    # Определим границы интервалов
    if i == 0:
        lower_bound = -np.inf  # Для первого интервала нижняя граница бесконечность
    else:
        lower_bound = (mid_points_1b[i - 1] + mid_points_1b[i]) / 2

    if i == len(mid_points_1b) - 1:
        upper_bound = np.inf  # Для последнего интервала верхняя граница бесконечность
    else:
        upper_bound = (mid_points_1b[i] + mid_points_1b[i + 1]) / 2

    # Используем нормальное распределение для расчета вероятности попадания в интервал
    prob = stats.norm.cdf(upper_bound, loc=mean_sample, scale=std_sample) - stats.norm.cdf(lower_bound, loc=mean_sample,
                                                                                           scale=std_sample)

    # Рассчитываем ожидаемую частоту для интервала
    expected_frequencies_1b.append(prob * total_count_1b)

expected_frequencies_1b = np.array(expected_frequencies_1b)

# Рассчитываем chi^2 для задания 1б
chi2_manual_1b = np.sum((observed_frequencies_1b - expected_frequencies_1b) ** 2 / expected_frequencies_1b)

# Шаг 2: Найдем критическое значение chi^2 для задания 1б
df_1b = len(observed_frequencies_1b) - 3  # Степени свободы = число групп - 1
print(f"Число групп: {len(observed_frequencies_1b)}")
chi2_critical_1b = stats.chi2.ppf(1 - alpha, df_1b)

# Вывод результатов
print(f"Наблюдаемые частоты для выборочных средних: {observed_frequencies_1b.values}")
print(f"Ожидаемые частоты для выборочных средних: {expected_frequencies_1b}")
print(f"Вычисленная chi^2 (Задание 1б): {chi2_manual_1b:.3f}")
print(f"Критическое значение chi^2 для df={df_1b} и уровня значимости 0.05 (Задание 1б): {chi2_critical_1b}")

# Сравнение chi^2 с критическим значением
if chi2_manual_1b > chi2_critical_1b:
    print("Нулевая гипотеза о нормальном распределении для выборочных средних отклоняется (Задание 1б).")
else:
    print("Нет оснований отклонять нулевую гипотезу о нормальном распределении для выборочных средних (Задание 1б).")
print(f" ")
print(f" ")

print(f"2")

# Параметры задачи
gamma = 0.95
delta = 3

# Стандартное отклонение выборки
sigma = np.std(crime_ages, ddof=0)

# Расчет объема выборки
z_value = stats.norm.ppf((1 + gamma) / 2)

def calculate_sample_size(sigma, delta, gamma, z_value):
    return (z_value * sigma / delta) ** 2

n = int(np.ceil(calculate_sample_size(sigma, delta, gamma, z_value)))

# Генерация выборочных средних для задачи 1б
sample_means = [np.mean(np.random.choice(crime_ages, size=n, replace=True)) for _ in range(36)]

# --- Часть 2: Проверка гипотезы о равенстве дисперсий ---
# Устанавливаем случайное начальное состояние для воспроизводимости результатов
random_state = 42
rng = np.random.default_rng(random_state)

# --- Проверка гипотезы о равенстве дисперсий для Задания 2а ---
# Альфа для 2а
alpha_2a = 0.05

# Генерируем две случайные выборки для 2а
sample1 = rng.choice(crime_ages, size=n, replace=True)
sample2 = rng.choice(crime_ages, size=n, replace=True)

# Вычисление дисперсий для каждой выборки
var1_2a = np.var(sample1, ddof=1)
var2_2a = np.var(sample2, ddof=1)

# Вывод дисперсий для каждой выборки
print(f"Дисперсия первой выборки (2а): {var1_2a}")
print(f"Дисперсия второй выборки (2а): {var2_2a}")

# Применение критерия Фишера для проверки гипотезы о равенстве дисперсий
f_stat_2a = var1_2a / var2_2a if var1_2a > var2_2a else var2_2a / var1_2a  # Рассчитываем F-статистику
df1_2a = len(sample1) - 1  # Степени свободы для первой выборки
df2_2a = len(sample2) - 1  # Степени свободы для второй выборки
p_value_f_2a = 1 - stats.f.cdf(f_stat_2a, df1_2a, df2_2a)  # Вычисляем p-value для F-критерия

# Критическое значение F для 2а
f_critical_2a = stats.f.ppf(1 - alpha_2a, df1_2a, df2_2a)

# Вывод результатов для 2а
print("\nКритерий Фишера для проверки равенства дисперсий (Задание 2а):")
print(f"F-статистика: {f_stat_2a}")
print(f"p-value: {p_value_f_2a}")
print(f"Критическое значение F: {f_critical_2a}")

# Принятие или отклонение гипотезы о равенстве дисперсий для 2а
if f_stat_2a > f_critical_2a:
    print("Нулевая гипотеза о равенстве дисперсий отклоняется (Задание 2а).")
else:
    print("Нет оснований отклонять нулевую гипотезу о равенстве дисперсий (Задание 2а).")

# --- Проверка гипотезы о равенстве дисперсий для Задания 2б ---
# Альфа для 2б
alpha_2b = 0.025

# Генерируем две новые случайные выборки для 2б
sample1_2b = rng.choice(crime_ages, size=n, replace=True)
sample2_2b = rng.choice(crime_ages, size=n, replace=True)

# Вычисление дисперсий для каждой выборки
var1_2b = np.var(sample1_2b, ddof=1)
var2_2b = np.var(sample2_2b, ddof=1)

# Вывод дисперсий для каждой выборки
print(f"\nДисперсия первой выборки (2б): {var1_2b}")
print(f"Дисперсия второй выборки (2б): {var2_2b}")

# Применение критерия Фишера для 2б
f_stat_2b = var1_2b / var2_2b if var1_2b > var2_2b else var2_2b / var1_2b  # Рассчитываем F-статистику
df1_2b = len(sample1_2b) - 1  # Степени свободы для первой выборки
df2_2b = len(sample2_2b) - 1  # Степени свободы для второй выборки
p_value_f_2b = 1 - stats.f.cdf(f_stat_2b, df1_2b, df2_2b)  # Вычисляем p-value для F-критерия

# Критическое значение F для 2б
f_critical_2b = stats.f.ppf(1 - alpha_2b, df1_2b, df2_2b)

# Вывод результатов для 2б
print("\nКритерий Фишера для проверки равенства дисперсий (Задание 2б):")
print(f"F-статистика: {f_stat_2b}")
print(f"p-value: {p_value_f_2b}")
print(f"Критическое значение F: {f_critical_2b}")

# Принятие или отклонение гипотезы о равенстве дисперсий для 2б
if f_stat_2b > f_critical_2b:
    print("Нулевая гипотеза о равенстве дисперсий отклоняется (Задание 2б).")
else:
    print("Нет оснований отклонять нулевую гипотезу о равенстве дисперсий (Задание 2б).")