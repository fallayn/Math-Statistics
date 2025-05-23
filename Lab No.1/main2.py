import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Выборка B
sample_B = np.array([
    -64, 5, -53, -29, -61, -49, -1, -22, -25, -38, -73, -20, -8, -37, -47,
    0, -37, -50, -46, -13, 7, -13, -42, -1, -44, -27, -20, -33, -37, -30,
    -20, -73, -57, -40, -4, -40, -83, -33, -37, -26, -79, -16, -77, -5, -51,
    -28, -63, -24, -25, -24, -38, 16, -37, -15, 29, -11, -14, -34, -31, -23,
    -16, -58, -73, -43, -31, -65, -12, -4, -38, -25, -31, -7, -9, -60, -61,
    -47, -46, -33, -15, -79, -48, 1, -62, -14, -49, -31, -25, -33, -38, -27,
    -51, -30, -43, -64, -24, -50, -22, -37, -6, -11, -78, -51, -1, -9, -34,
    1, -17, -33, 11, -54, -31, -34, -38, -22, -2, -9, -15, -6, -87, -45,
    -22, -30, -15, -30, -18, -77, 6, -47, -33, -21, -86, -31, -45, -43, -19,
    -36, -46, -69, -22, -59, -30, -22, 5, -29, -42, -47, 5, -17, -71, -36,
    6, -6, -7, -41, -37, -11, -11, -65, -36, -58, -36, -30, -46, -15, -49,
    -88, -12, -8, -83, -13, -30, -48, -66, -9, -31, -13, -32, -21, -47, -50,
    -25, -6, -31, -75, -48, -77, -13, -55, -26, -9, -32, -41, -68, -55, -53,
    25, -77, 1, -65, -35, -51, -24, -42
])

# Пункт 2.1
# Максимальный и минимальный элементы
x_max = np.max(sample_B)
x_min = np.min(sample_B)
R = x_max - x_min

# Оптимальное количество интервалов (Стерджес)
n = len(sample_B)
k = 12  # Ручная корректировка
h = R / k  # Длина интервала

# Создание интервалов
bins = np.linspace(x_min - h/2, x_max + h/2, k + 1)
intervals = [(round(bins[i], 2), round(bins[i+1], 2)) for i in range(len(bins)-1)]

# Подсчет частот
freq, _ = np.histogram(sample_B, bins=bins)

# Построение гистограммы и полигона
plt.figure(figsize=(12, 6))
plt.hist(sample_B, bins=bins, edgecolor='black', alpha=0.7, label='Гистограмма')
midpoints = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
plt.plot(midpoints, freq, marker='o', linestyle='-', color='r', label='Полигон')
plt.xlabel('Интервалы')
plt.ylabel('Частота')
plt.title('Гистограмма и полигон частот выборки B')
plt.legend()
plt.grid(linestyle='--')
plt.show()

# Интервалы (границы) и значения F(x)
edges = [-np.inf, -92.88, -82.31, -71.75, -61.19, -50.62, -40.06,
         -29.50, -18.94, -8.38, 2.19, 12.75, 23.31, np.inf]
values = [0.0000, 0.025, 0.079, 0.133, 0.212, 0.355,
          0.577, 0.720, 0.858, 0.952, 0.986, 0.991, 1.000]

# Построение ЭФР
plt.figure(figsize=(10, 6))

for i in range(len(values)):
    x_start = edges[i]
    x_end = edges[i + 1]
    y = values[i]

    if np.isinf(x_start):
        x_start = edges[i + 1] - 1.5  # условный старт для визуализации на графике
    if np.isinf(x_end):
        x_end = edges[i] + 1.5  # условный конец для последнего сегмента

    # Горизонтальный отрезок
    plt.hlines(y, xmin=x_start, xmax=x_end, colors='blue', linewidth=1.5)

    # Стрелка в начале отрезка (слева направо)
    plt.annotate(
        '',
        xy=(x_start + 0.3, y),
        xytext=(x_start, y),
        arrowprops=dict(arrowstyle='<-', color='blue', lw=1.5),
        annotation_clip=False
    )

# Кумулята по правым концам интервалов
right_edges = [e for e in edges[1:-1] if not np.isinf(e)] + [edges[-2]]
plt.plot(right_edges, values, linestyle='--', color='red', label='Кумулята')

# Настройка графика
plt.xlim(-100, 30)
plt.ylim(-0.05, 1.05)
plt.xlabel('Значения выборки')
plt.ylabel('F(x)')
plt.title('Эмпирическая функция распределения и кумулята выборки B')
plt.grid(linestyle='--', alpha=0.5)
plt.legend()
plt.show()

# Вывод таблицы интервалов
print("Интервальный ряд и частоты:")
print("Интервал\t\tКоличество элементов")
for interval, count in zip(intervals, freq):
    print(f"[{interval[0]:.2f}, {interval[1]:.2f})\t\t{count}")


# === Часть 1. Выборка A: числовые характеристики ===
# (предполагаем, что sample_B определена где-то выше)
n_A = len(sample_B)

# Среднее выборочное
mean_A = np.mean(sample_B)

# Медиана
median_A = np.median(sample_B)

# Мода (самое частое значение)
freqs_A = Counter(sample_B)
mode_A = freqs_A.most_common(1)[0][0]

# Размах
R_A = np.max(sample_B) - np.min(sample_B)

# Среднее абсолютное отклонение
mad_A = np.mean(np.abs(sample_B - mean_A))

print(np.sum(np.abs(sample_B - mean_A)))

# Относительное линейное отклонение
K_d_A = mad_A / abs(mean_A)

# Коэффициент осцилляции
K_o_A = R_A / abs(mean_A) * 100

# Дисперсия (смещённая) и СКО
var_A = np.var(sample_B, ddof=0)
std_A = np.sqrt(var_A)

# Несмещённая дисперсия и СКО
unbiased_var_A = np.var(sample_B, ddof=1)
unbiased_std_A = np.sqrt(unbiased_var_A)

# Коэффициент вариации
V_A = unbiased_std_A / abs(mean_A) * 100

# Вывод результатов
print("Выборка A — числовые характеристики:")
print(f"n = {n_A}")
print(f"Среднее выборочное       = {mean_A:.4f}")
print(f"Медиана                  = {median_A:.4f}")
print(f"Мода                     = {mode_A}")
print(f"Размах (R)               = {R_A:.4f}")
print(f"Среднее абсолют. отклон. = {mad_A:.4f}")
print(f"Относит. лин. отклон.    = {K_d_A:.4f}")
print(f"Коэфф. осцилляции (Ko)   = {K_o_A:.2f}%")
print(f"Дисперсия (смещ.)        = {var_A:.4f}")
print(f"СКО (смещ.)              = {std_A:.4f}")
print(f"Дисперсия (несмещ.)     = {unbiased_var_A:.4f}")
print(f"СКО (несмещ.)           = {unbiased_std_A:.4f}")
print(f"Коэффициент вариации V   = {V_A:.2f}%")
print()

# === Часть 2. Выборка B: гистограмма и полигон относительных частот ===
# (предполагаем, что sample_B определена)
n_B = len(sample_B)

# Выбор числа интервалов и сами бины
k = int(np.ceil(1 + np.log2(n_B)))  # Формула Стерджеса
counts, bins = np.histogram(sample_B, bins=k)

# Переводим в относительные частоты
rel_freq = counts / n_B

# Средние точки интервалов
midpoints = (bins[:-1] + bins[1:]) / 2

plt.figure(figsize=(10,6))
# Гистограмма относительных частот
plt.bar(midpoints, rel_freq, 
        width=(bins[1]-bins[0])*0.9, 
        edgecolor='black', alpha=0.6, 
        label='Гистограмма (отн. частоты)')

# Полигон относительных частот
plt.plot(midpoints, rel_freq, marker='o', linestyle='-', color='red', 
         label='Полигон (отн. частоты)')

plt.xlabel('Значение')
plt.ylabel('Относительная частота')
plt.title('Гистограмма и полигон относит. частот выборки B')
plt.grid(linestyle='--', alpha=0.5)
plt.legend()
plt.show()
