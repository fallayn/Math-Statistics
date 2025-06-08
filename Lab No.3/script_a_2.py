import numpy as np
from scipy import stats

# Выборка A
data_a = np.array([
    7, 11, 5, 5, 5, 5, 9, 4, 5, 3, 8, 5, 3, 8, 3, 11, 3, 9, 6, 8, 3, 3, 6, 2, 7, 4, 4, 3, 5, 7, 4, 6, 5, 2, 9, 5, 8,
    6, 1, 1, 7, 7, 4, 4, 9, 7, 4, 3, 1, 6, 6, 4, 5, 4, 5, 5, 7, 8, 6, 8, 4, 10, 2, 7, 7, 5, 9, 6, 11, 2, 7, 7, 9, 2,
    6, 8
])

# Параметры
alpha = 0.05
n = len(data_a)

# =============================================================================
# 1.3 Проверка гипотезы о среднем (H0: a = a0)
# =============================================================================

# Вычисляем статистики
mean_a = np.mean(data_a)
var_a = np.var(data_a, ddof=1)  # исправленная дисперсия
std_a = np.sqrt(var_a)

# a0 = [x̄] + 0.5
a0 = np.floor(mean_a) + 0.5

# Z-статистика
z_stat = (mean_a - a0) / (std_a / np.sqrt(n))

# Критическое значение (двусторонний тест)
z_critical = stats.norm.ppf(1 - alpha/2)

# Решение
reject_H0_mean = np.abs(z_stat) > z_critical

print("="*80)
print("1.3 Проверка гипотезы о среднем значении")
print("="*80)
print(f"Выборочное среднее (x̄): {mean_a:.6f}")
print(f"Исправленное стандартное отклонение (s): {std_a:.6f}")
print(f"a0 = [x̄] + 0.5 = [{np.floor(mean_a)}] + 0.5 = {a0:.1f}")
print(f"Z-статистика: {z_stat:.6f}")
print(f"Критическое значение (α={alpha}): ±{z_critical:.6f}")
print(f"Гипотеза H0 {'отвергается' if reject_H0_mean else 'не отвергается'}")
print()

# =============================================================================
# 1.4 Проверка гипотезы о дисперсии (H0: σ² = σ0²)
# =============================================================================

# σ0² = [s²] + 0.5
sigma0_sq = np.floor(var_a) + 0.5

# χ²-статистика
chi2_stat = (n - 1) * var_a / sigma0_sq

# Критические значения
df = n - 1  # степени свободы
chi2_lower = stats.chi2.ppf(alpha/2, df)
chi2_upper = stats.chi2.ppf(1 - alpha/2, df)

# Решение
reject_H0_var = chi2_stat < chi2_lower or chi2_stat > chi2_upper

print("="*80)
print("1.4 Проверка гипотезы о дисперсии")
print("="*80)
print(f"Исправленная дисперсия (s²): {var_a:.6f}")
print(f"σ0² = [s²] + 0.5 = [{np.floor(var_a)}] + 0.5 = {sigma0_sq:.1f}")
print(f"χ²-статистика: {chi2_stat:.6f}")
print(f"Критические значения (α={alpha}): ({chi2_lower:.6f}, {chi2_upper:.6f})")
print(f"Гипотеза H0 {'отвергается' if reject_H0_var else 'не отвергается'}")
print()

# =============================================================================
# 1.5 Проверка гипотезы о равенстве средних в двух подвыборках
# =============================================================================

# Разбиваем выборку на две части
n_half = n // 2
sample1 = data_a[:n_half]
sample2 = data_a[n_half:]

# Вычисляем статистики для подвыборок
mean1 = np.mean(sample1)
mean2 = np.mean(sample2)
var1 = np.var(sample1, ddof=1)
var2 = np.var(sample2, ddof=1)
n1 = len(sample1)
n2 = len(sample2)

# Проверка равенства дисперсий (F-тест)
F_stat = var1 / var2 if var1 >= var2 else var2 / var1
df1 = n1 - 1
df2 = n2 - 1
F_critical = stats.f.ppf(1 - alpha/2, df1, df2)
equal_variances = F_stat < F_critical

# Проверка гипотезы о равенстве средних
if equal_variances:
    # t-тест с равными дисперсиями
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    t_stat = (mean1 - mean2) / np.sqrt(pooled_var * (1/n1 + 1/n2))
    df_t = n1 + n2 - 2
    t_critical = stats.t.ppf(1 - alpha/2, df_t)
    reject_H0_means = np.abs(t_stat) > t_critical
    test_type = "t-тест с равными дисперсиями"
else:
    # t-тест с неравными дисперсиями (Уэлча)
    t_stat = (mean1 - mean2) / np.sqrt(var1/n1 + var2/n2)
    df_t = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    t_critical = stats.t.ppf(1 - alpha/2, df_t)
    reject_H0_means = np.abs(t_stat) > t_critical
    test_type = "t-тест Уэлча (неравные дисперсии)"

print("="*80)
print("1.5 Проверка гипотезы о равенстве средних в двух подвыборках")
print("="*80)
print(f"Подвыборка 1: n={n1}, x̄={mean1:.6f}, s²={var1:.6f}")
print(f"Подвыборка 2: n={n2}, x̄={mean2:.6f}, s²={var2:.6f}")

print("\nПроверка равенства дисперсий (F-тест):")
print(f"F-статистика = {F_stat:.6f}")
print(f"Критическое значение (α={alpha}): {F_critical:.6f}")
print(f"Гипотеза о равенстве дисперсий {'не отвергается' if equal_variances else 'отвергается'}")

print(f"\nПроверка равенства средних ({test_type}):")
print(f"t-статистика: {t_stat:.6f}")
print(f"Степени свободы: {df_t:.2f}")
print(f"Критическое значение (α={alpha}): ±{t_critical:.6f}")
print(f"Гипотеза H0 {'отвергается' if reject_H0_means else 'не отвергается'}")
print()

# =============================================================================
# Сводка результатов
# =============================================================================
print("="*80)
print("СВОДКА РЕЗУЛЬТАТОВ ДЛЯ ВЫБОРКИ A")
print("="*80)
print(f"1.3 Гипотеза о среднем (a = {a0:.1f}): {'Отвергается' if reject_H0_mean else 'Не отвергается'}")
print(f"1.4 Гипотеза о дисперсии (σ² = {sigma0_sq:.1f}): {'Отвергается' if reject_H0_var else 'Не отвергается'}")
print(f"1.5 Гипотеза о равенстве средних: {'Отвергается' if reject_H0_means else 'Не отвергается'}")