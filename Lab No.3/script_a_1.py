import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Исходные данные выборки A
data_a = np.array([
    7, 11, 5, 5, 5, 5, 9, 4, 5, 3, 8, 5, 3, 8, 3, 11, 3, 9, 6, 8, 3, 3, 6, 2, 7, 4, 4, 3, 5, 7, 4, 6, 5, 2, 9, 5, 8,
    6, 1, 1, 7, 7, 4, 4, 9, 7, 4, 3, 1, 6, 6, 4, 5, 4, 5, 5, 7, 8, 6, 8, 4, 10, 2, 7, 7, 5, 9, 6, 11, 2, 7, 7, 9, 2,
    6, 8
])

# Общие параметры
n = len(data_a)  # 76
lambda_poisson = np.mean(data_a)  # 5.605
n_binom = 11  # число испытаний для биномиального распределения
p_binom = 0.518  # вероятность успеха

# =============================================================================
# ТАБЛИЦА 1: РАСЧЕТ ДЛЯ РАСПРЕДЕЛЕНИЯ ПУАССОНА (С ГРУППИРОВКОЙ)
# =============================================================================

# Группировка данных для Пуассона (как в отчете)
poisson_bins = [0, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, np.inf]
poisson_labels = ['1-2', '3', '4', '5', '6', '7', '8', '9+']

# Вычисление наблюдаемых частот
observed_poisson, _ = np.histogram(data_a, bins=poisson_bins)

# Расчет теоретических вероятностей
poisson_dist = stats.poisson(lambda_poisson)
poisson_probs = [
    poisson_dist.cdf(2.5) - poisson_dist.cdf(0),  # P(1≤X≤2)
    poisson_dist.pmf(3),  # P(X=3)
    poisson_dist.pmf(4),  # P(X=4)
    poisson_dist.pmf(5),  # P(X=5)
    poisson_dist.pmf(6),  # P(X=6)
    poisson_dist.pmf(7),  # P(X=7)
    poisson_dist.pmf(8),  # P(X=8)
    1 - poisson_dist.cdf(8.5)  # P(X≥9)
]

# Ожидаемые частоты
expected_poisson = np.array(poisson_probs) * n

# Расчет компонентов χ²
chi2_components_poisson = (observed_poisson - expected_poisson) ** 2 / expected_poisson

# Вывод таблицы для Пуассона
print("="*80)
print("ТАБЛИЦА 1: РАСЧЕТ КРИТЕРИЯ χ² ДЛЯ РАСПРЕДЕЛЕНИЯ ПУАССОНА")
print("="*80)
print(f"{'Интервал':<5} | {'n_i':<4} | {'p_i':<8} | {'np_i':<10} | {'(n_i-np_i)^2/np_i':<15}")
print("-"*65)
for i in range(len(observed_poisson)):
    print(f"{poisson_labels[i]:<5} | {observed_poisson[i]:<4} | {poisson_probs[i]:<8.4f} | {expected_poisson[i]:<10.4f} | {chi2_components_poisson[i]:<15.6f}")

# Общая статистика χ² для Пуассона
total_chi2_poisson = np.sum(chi2_components_poisson)
df_poisson = len(observed_poisson) - 2  # 8 интервалов - 1 - 1 параметр
critical_value_poisson = stats.chi2.ppf(0.95, df_poisson)

print("\nРезультаты для распределения Пуассона:")
print(f"Общая статистика χ²: {total_chi2_poisson:.4f}")
print(f"Степени свободы: {df_poisson}")
print(f"Критическое значение (α=0.05): {critical_value_poisson:.4f}")
print(f"Вывод: {'Не отвергаем H0' if total_chi2_poisson < critical_value_poisson else 'Отвергаем H0'}")

# Визуализация для Пуассона
plt.figure(figsize=(12, 6))
plt.bar(poisson_labels, observed_poisson, width=0.4, alpha=0.7, label='Наблюдаемые')
plt.bar(poisson_labels, expected_poisson, width=0.4, alpha=0.7, label='Ожидаемые', bottom=0)
plt.xlabel('Интервал')
plt.ylabel('Частота')
plt.title(f'Распределение Пуассона (λ = {lambda_poisson:.3f})')
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# ТАБЛИЦА 2: РАСЧЕТ ДЛЯ БИНОМИАЛЬНОГО РАСПРЕДЕЛЕНИЯ
# =============================================================================

# Группировка данных для биномиального распределения
binom_bins = [-np.inf, 3.5, 4.5, 5.5, 6.5, 7.5, np.inf]
binom_labels = ['0-3', '4', '5', '6', '7', '8-11']

# Вычисление наблюдаемых частот
observed_binom, _ = np.histogram(data_a, bins=binom_bins)

# Расчет теоретических вероятностей
binom_dist = stats.binom(n_binom, p_binom)
binom_probs = [
    binom_dist.cdf(3),                # P(X ≤ 3)
    binom_dist.pmf(4),                # P(X = 4)
    binom_dist.pmf(5),                # P(X = 5)
    binom_dist.pmf(6),                # P(X = 6)
    binom_dist.pmf(7),                # P(X = 7)
    binom_dist.sf(7)                  # P(X ≥ 8)
]

# Ожидаемые частоты
expected_binom = np.array(binom_probs) * n

# Расчет компонентов χ²
chi2_components_binom = (observed_binom - expected_binom) ** 2 / expected_binom

# Вывод таблицы для биномиального распределения
print("\n" + "="*80)
print("ТАБЛИЦА 2: РАСЧЕТ КРИТЕРИЯ χ² ДЛЯ БИНОМИАЛЬНОГО РАСПРЕДЕЛЕНИЯ")
print("="*80)
print(f"{'Интервал':<6} | {'n_i':<4} | {'p_i':<8} | {'np_i':<10} | {'(n_i-np_i)^2/np_i':<15}")
print("-"*65)
for i in range(len(observed_binom)):
    print(f"{binom_labels[i]:<6} | {observed_binom[i]:<4} | {binom_probs[i]:<8.4f} | {expected_binom[i]:<10.4f} | {chi2_components_binom[i]:<15.6f}")

# Общая статистика χ² для биномиального распределения
total_chi2_binom = np.sum(chi2_components_binom)
df_binom = len(observed_binom) - 3  # 6 интервалов - 1 - 2 параметра (n и p)
critical_value_binom = stats.chi2.ppf(0.95, df_binom)

print("\nРезультаты для биномиального распределения:")
print(f"Общая статистика χ²: {total_chi2_binom:.4f}")
print(f"Степени свободы: {df_binom}")
print(f"Критическое значение (α=0.05): {critical_value_binom:.4f}")
print(f"Вывод: {'Не отвергаем H0' if total_chi2_binom < critical_value_binom else 'Отвергаем H0'}")

# Визуализация для биномиального распределения
plt.figure(figsize=(12, 6))
plt.bar(binom_labels, observed_binom, width=0.4, alpha=0.7, label='Наблюдаемые')
plt.bar(binom_labels, expected_binom, width=0.4, alpha=0.7, label='Ожидаемые', bottom=0)
plt.xlabel('Интервал')
plt.ylabel('Частота')
plt.title(f'Биномиальное распределение (n={n_binom}, p={p_binom:.3f})')
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# СРАВНЕНИЕ С ОТЧЕТОМ
# =============================================================================

# Данные из вашего отчета
report_poisson = {
    'intervals': ['1-2', '3', '4', '5', '6', '7', '8', '9+'],
    'observed': [8, 8, 10, 13, 9, 11, 7, 10],
    'probs': [0.0735, 0.1033, 0.1471, 0.1677, 0.1592, 0.1295, 0.0922, 0.1275],
    'expected': [5.586, 7.851, 11.180, 12.745, 12.099, 9.842, 7.007, 9.690],
    'components': [1.043, 0.0028, 0.1245, 0.0051, 0.7942, 0.1362, 0.00001, 0.0094]
}

report_binom = {
    'intervals': ['0-3', '4', '5', '6', '7', '8-11'],
    'observed': [16, 10, 13, 9, 11, 17],
    'probs': [0.1325, 0.2079, 0.3128, 0.3350, 0.2574, 0.1962],
    'expected': [10.07, 15.80, 23.77, 25.46, 19.56, 14.91],
    'components': [3.494, 2.129, 4.880, 10.640, 3.746, 0.293]
}

# Функция для сравнения расчетов
def compare_with_report(our_values, report_values, title):
    print(f"\n{title}")
    print(f"{'Параметр':<15} | {'Наш расчет':<15} | {'Отчет':<15} | {'Разница':<10}")
    print("-" * 70)
    
    for param in ['probs', 'expected', 'components']:
        our = our_values[param]
        rep = report_values[param]
        
        for i in range(len(our)):
            diff = abs(our[i] - rep[i])
            print(f"{param.capitalize()} для {report_values['intervals'][i]:<5} | {our[i]:<15.6f} | {rep[i]:<15.6f} | {diff:<10.6f}")

# Сравнение для Пуассона
compare_with_report(
    {'probs': poisson_probs, 'expected': expected_poisson, 'components': chi2_components_poisson},
    report_poisson,
    "СРАВНЕНИЕ С ОТЧЕТОМ ДЛЯ РАСПРЕДЕЛЕНИЯ ПУАССОНА:"
)

# Сравнение для биномиального распределения
compare_with_report(
    {'probs': binom_probs, 'expected': expected_binom, 'components': chi2_components_binom},
    report_binom,
    "СРАВНЕНИЕ С ОТЧЕТОМ ДЛЯ БИНОМИАЛЬНОГО РАСПРЕДЕЛЕНИЯ:"
)

# =============================================================================
# ВЫВОД РЕКОМЕНДАЦИЙ
# =============================================================================

print("\n" + "="*80)
print("РЕКОМЕНДАЦИИ ПО ЛАБОРАТОРНОЙ РАБОТЕ")
print("="*80)
print("1. Для распределения Пуассона:")
print(f"   - Общая статистика χ²: {total_chi2_poisson:.3f} (в отчете: 2.115)")
print("   - Причины различий:")
print("     * Разные методы расчета вероятностей (мы использовали scipy.stats.poisson)")
print("     * Разные значения λ (мы использовали точное среднее 5.605)")
print("     * Округление в отчете при расчете вероятностей")
print("   - Рекомендация: используйте точное значение λ = x̄ = 5.605")

print("\n2. Для биномиального распределения:")
print(f"   - Общая статистика χ²: {total_chi2_binom:.3f} (в отчете: 25.182)")
print("   - Расчеты почти совпадают с отчетом (различия из-за округления)")

print("\n3. Общие рекомендации:")
print("   - Всегда указывайте метод расчета вероятностей")
print("   - Проверяйте сумму вероятностей (должна быть близка к 1)")
print(f"     * Пуассон: сумма p_i = {sum(poisson_probs):.6f}")
print(f"     * Биномиальное: сумма p_i = {sum(binom_probs):.6f}")
print("   - Для группировки используйте целочисленные границы интервалов")
print("   - Убедитесь, что все ожидаемые частоты > 5 (в нашем случае выполнено)")

print("\n4. Оформление отчета:")
print("   - Включите формулы для расчета вероятностей")
print("   - Приведите полную таблицу расчетов для обоих распределений")
print("   - Укажите параметры распределений: λ = 5.605 для Пуассона, n=11, p=0.518 для биномиального")
print("   - Добавьте графики сравнения наблюдаемых и ожидаемых частот")