import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Выборка A
sample_A = np.array([
    7, 11, 5, 5, 5, 5, 9, 4, 5, 3, 8, 5, 3, 8, 3,
    11, 3, 9, 6, 8, 3, 3, 6, 2, 7, 4, 4, 3, 5, 7,
    4, 6, 5, 2, 9, 5, 8, 6, 1, 1, 7, 7, 4, 4, 9,
    7, 4, 3, 1, 6, 6, 4, 5, 4, 5, 5, 7, 8, 6, 8,
    4, 10, 2, 7, 7, 5, 9, 6, 11, 2, 7, 7, 9, 2, 6, 8
])

# 1.1 Максимальный, минимальный элементы и размах
max_A = np.max(sample_A)
min_A = np.min(sample_A)
range_A = max_A - min_A

# 1.2 Статистический ряд и полигон относительных частот
values, counts = np.unique(sample_A, return_counts=True)
n = len(sample_A)
relative_freq = counts / n  # Относительные частоты

# Построение полигона относительных частот
plt.figure(figsize=(8, 5))
plt.plot(values, relative_freq, marker='o', linestyle='-', color='r', label="Полигон относительных частот")
plt.xlabel("Значения выборки")
plt.ylabel("Относительная частота (w_i)")
plt.title("Полигон относительных частот выборки A")
plt.grid()
plt.legend()
plt.show()

# 1.3 Эмпирическая функция распределения (исправленная версия)
sorted_values = np.sort(sample_A)
n = len(sorted_values)
unique_values, indices = np.unique(sorted_values, return_index=True)

# Корректный расчет накопленных частот (учитываем все значения)
counts = np.bincount(sorted_values)[unique_values]
max_cdf = np.cumsum(counts) / n  # Накопленные частоты

# Преобразуем unique_values в float
unique_values = unique_values.astype(float)

# Добавляем начальную точку (-inf, 0) и конечную точку (max_A + 1, 1)
extended_values = np.insert(unique_values, 0, -np.inf)
extended_values = np.append(extended_values, max_A + 1)
extended_cdf = np.insert(max_cdf, 0, 0)
extended_cdf = np.append(extended_cdf, 1.0)

plt.figure(figsize=(8, 5))

# Рисуем горизонтальные линии между точками разрыва
for i in range(len(extended_values) - 1):
    plt.hlines(
        y=extended_cdf[i],
        xmin=extended_values[i],
        xmax=extended_values[i+1],
        colors='b',
        linestyle='-',
        linewidth=1.5
    )

# Добавляем стрелки слева от точек разрыва (включая x=1)
for x, y in zip(unique_values, max_cdf):
    plt.annotate(
        '', 
        xy=(x, y), 
        xytext=(x+0.2, y),
        arrowprops=dict(
            arrowstyle='->', 
            color='b', 
            lw=1.5, 
            shrinkA=0, 
            shrinkB=0
        ),
        annotation_clip=False
    )

# Добавляем стрелку для начального сегмента (x <= 1)
plt.hlines(
    y=0,
    xmin=0,
    xmax=1,
    colors='b',
    linestyle='-',
    linewidth=1.5
)
plt.annotate(
    '', 
    xy=(0, 0), 
    xytext=(0.2, 0), 
    arrowprops=dict(arrowstyle='->', color='b', lw=1.5),
    annotation_clip=False
)

# Настраиваем оси
plt.xlim(min_A - 1, max_A + 1)
plt.ylim(-0.05, 1.05)
plt.xlabel("Значения выборки")
plt.ylabel("F(x)")
plt.title("Эмпирическая функция распределения выборки A")
plt.grid(linestyle='--', alpha=0.5)
plt.show()

# 1.4 Числовые характеристики положения
mean_A = np.mean(sample_A)  # Среднее выборочное
median_A = np.median(sample_A)  # Медиана
mode_A = stats.mode(sample_A)[0]  # Мода

# Дисперсия и стандартное отклонение
variance_A = np.var(sample_A, ddof=1)  # Выборочная дисперсия
nm_variance_A = np.var(sample_A, ddof=0) # Дисперсия
std_A = np.std(sample_A, ddof=1)  # Стандартное отклонение
nm_std_a = np.std(sample_A, ddof=0) # Стандартное отклонение для НЕ выборочной дисперсии

# Среднее арифметическое абсолютных отклонений (ваша формула)
unique_values, counts = np.unique(sample_A, return_counts=True)
abs_deviations_weighted = np.sum(np.abs(unique_values - mean_A) * counts)
d_bar = abs_deviations_weighted / len(sample_A)

# MAD (существующий расчет для проверки)
mad = np.mean(np.abs(sample_A - mean_A)) 

# 1.5 Центральные моменты 3-го и 4-го порядка
m3_A = np.mean((sample_A - mean_A) ** 3)
m4_A = np.mean((sample_A - mean_A) ** 4)

#1.4 Начальные моменты 3-го и 4-го порядка
s3_A = np.mean(sample_A ** 3)
s4_A = np.mean(sample_A ** 4)

# Коэффициенты асимметрии и эксцесса
skewness_A = stats.skew(sample_A)
kurtosis_A = stats.kurtosis(sample_A, fisher=True)

# Вывод результатов
print("Числовые характеристики выборки:")
print("-" * 50)
print(f"{'Максимум:':<25}{max_A:>10.4f}")
print(f"{'Минимум:':<25}{min_A:>10.4f}")
print(f"{'Размах:':<25}{range_A:>10.4f}")
print(f"{'Среднее:':<25}{mean_A:>10.4f}")
print(f"{'Медиана:':<25}{median_A:>10.4f}")
print(f"{'Мода:':<25}{mode_A:>10}")
print(f"{'Несмещенная дисперсия:':<25}{variance_A:>10.4f}")
print(f"{'Дисперсия':<25}{nm_variance_A:>10.4f}")
print(f"{'Стандартное несмещенное отклонение:':<25}{std_A:>10.4f}")
print(f"{'Стандартное отклонение':<25}{nm_std_a:>10.4f}")
print(f"{'Знаменатель для ср. абс. отклонений':<25}{abs_deviations_weighted:>10.4f}")
print(f"{'Ср. абс. отклонений (d̄):':<25}{d_bar:>10.4f}")  
print(f"{'MAD (проверка):':<25}{mad:>10.4f}")
print(f"{'Асимметрия:':<25}{skewness_A:>10.4f}")
print(f"{'Эксцесс:':<25}{kurtosis_A:>10.4f}")
print("-" * 50)
print("Моменты:")
print(f"{'Центральный момент 3-го порядка:':<35}{m3_A:>10.4f}")
print(f"{'Центральный момент 4-го порядка:':<35}{m4_A:>10.4f}")
print(f"{'Начальный момент 3-го порядка:':<35}{s3_A:>10.4f}")
print(f"{'Начальный момент 4-го порядка:':<35}{s4_A:>10.4f}")
