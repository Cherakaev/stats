import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math
import seaborn as sns


# 30 вариант


def absolute_frequencies(x: list[float]) -> tuple:
    count_bins, bound_bins = np.histogram(x, bins='fd')
    print("Интервальный ряд абсолютных частот(частоты и их границы): ")
    print(count_bins)
    print(bound_bins)
    print("")
    sum_freq = sum(count_bins)
    print("Сумма частот интервального ряда: ")
    print(sum_freq)
    print("\n")
    return count_bins, bound_bins


def relative_frequencies(count_bins: np.ndarray[float], n: int) -> np.ndarray[float]:
    rel_count_bins = count_bins / n
    sum_freq = sum(rel_count_bins)
    print("Интервальный ряд относительных частот(частоты): ")
    print(rel_count_bins.tolist())
    print("")
    print("Сумма частот интервального ряда: ")
    print(sum_freq)
    print("\n")
    return rel_count_bins


def plot_many_hist(x: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for bins in range(2, 11):
        axes[0].hist(x, bins=bins, density=True, alpha=0.5, label=f"{bins} интервалов")
    axes[0].set_title("Гистограммы относительных частот (число интервалов 2–10)")
    axes[0].set_xlabel("Значения X")
    axes[0].set_ylabel("Относительная частота")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    for bins in range(15, 26):
        axes[1].hist(x, bins=bins, density=True, alpha=0.5, label=f"{bins} интервалов")
    axes[1].set_title("Гистограммы относительных частот (число интервалов 15–25)")
    axes[1].set_xlabel("Значения X")
    axes[1].set_ylabel("Относительная частота")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_abs_hist(x: np.ndarray) -> None:
    plt.figure()
    plt.hist(x, bins='fd', color="green", edgecolor="black")
    plt.title("Гистограмма абсолютных частот")
    plt.xlabel("Значения X")
    plt.ylabel("Частота")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def plot_rel_hist(x: np.ndarray, a: int, sigma: int) -> None:
    plt.figure()
    plt.hist(x, bins='fd', density=True, color="skyblue", edgecolor="black")

    x_range = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_range, stats.norm.pdf(x_range, a, sigma),
             'r-', linewidth=2, label='Теоретическая кривая')

    plt.title("Гистограмма относительных частот")
    plt.xlabel("Значения X")
    plt.ylabel("Частота")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_cumulate(x: np.ndarray, a: int, sigma: int) -> None:
    plt.figure()
    plt.hist(x, bins='fd', density=True, cumulative=True, edgecolor="black",
             label="Эмпирическая функция распределения",  histtype='step')

    x_range = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_range, stats.norm.cdf(x_range, a, sigma),
             'r-', linewidth=2, label='Теоретическая функция распределения')

    plt.title("Графики функций распределения")
    plt.xlabel("Значения X")
    plt.ylabel("Частота")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def plot_boxplot(x: np.ndarray) -> None:
    q1 = np.percentile(x, 25)
    q2 = np.percentile(x, 50)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1

    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr

    lower_whisker_actual = np.min(x[x >= lower_whisker])
    upper_whisker_actual = np.max(x[x <= upper_whisker])

    outliers = x[(x < lower_whisker) | (x > upper_whisker)]

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x, color="lightblue")
    plt.title("Бокс-плот распределения")
    plt.xlabel("Значения X")

    print("=" * 50)
    print("СТАТИСТИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ БОКС-ПЛОТА")
    print("=" * 50)
    print(f"Первый квартиль (Q1, 25%): {q1:.4f}")
    print(f"Медиана (Q2, 50%): {q2:.4f}")
    print(f"Третий квартиль (Q3, 75%): {q3:.4f}")
    print(f"Интерквартильный размах (IQR): {iqr:.4f}")
    print(f"Теоретическая нижняя граница уса: {lower_whisker:.4f}")
    print(f"Теоретическая верхняя граница уса: {upper_whisker:.4f}")
    print(f"Фактическая нижняя граница уса: {lower_whisker_actual:.4f}")
    print(f"Фактическая верхняя граница уса: {upper_whisker_actual:.4f}")
    print(f"Выбросы: {outliers}")
    print(f"Количество выбросов: {len(outliers)}")

    if abs(q2 - q1 - (q3 - q2)) / iqr < 0.1:
        print("- Распределение примерно симметрично")
    elif q2 - q1 > q3 - q2:
        print("- Распределение скошено влево (длинный левый хвост)")
    else:
        print("- Распределение скошено вправо (длинный правый хвост)")

    plt.show()


def manual_stats(x: np.ndarray) -> dict:
    n = len(x)
    avg = sum(x) / n

    x_sorted = sorted(x)
    if n % 2 == 1:
        median = x_sorted[n // 2]
    else:
        median = (x_sorted[n // 2 - 1] + x_sorted[n // 2]) / 2

    mode_manual = mode(x)

    distorted_dispersion = sum((xi - avg) ** 2 for xi in x) / n
    undistorted_dispersion = sum((xi - avg) ** 2 for xi in x) / (n-1)

    distorted_std = math.sqrt(distorted_dispersion)
    undistorted_std = math.sqrt(undistorted_dispersion)

    skewness = sum((xi - avg) ** 3 for xi in x) / (n * distorted_std ** 3)
    kurtosis = sum((xi - avg) ** 4 for xi in x) / (n * distorted_std ** 4) - 3

    return {
        'mean': avg,
        'median': median,
        'mode_manual': mode_manual,
        'distorted_dispersion': distorted_dispersion,
        'undistorted_dispersion': undistorted_dispersion,
        'distorted_std': distorted_std,
        'undistorted_std': undistorted_std,
        'skewness': skewness,
        'kurtosis': kurtosis
    }


def library_stats(x):

    avg = np.mean(x)
    median = np.median(x)

    try:
        mode_lib = stats.mode(x)[0][0]
    except:
        hist, bin_edges = np.histogram(x, bins='auto')
        mode_idx = np.argmax(hist)
        mode_lib = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2

    distorted_dispersion = np.var(x)
    undistorted_dispersion = np.var(x, ddof=1)

    distorted_std = np.std(x)
    undistorted_std = np.std(x, ddof=1)

    skewness= stats.skew(x, bias=False)  # несмещенная оценка
    kurtosis = stats.kurtosis(x, bias=False)  # эксцесс (уже с вычитанием 3)

    return {
        'mean': avg,
        'median': median,
        'mode_manual': mode_lib,
        'distorted_dispersion': distorted_dispersion,
        'undistorted_dispersion': undistorted_dispersion,
        'distorted_std': distorted_std,
        'undistorted_std': undistorted_std,
        'skewness': skewness,
        'kurtosis': kurtosis
    }

def mode(x: np.ndarray) -> float:
    frequency = {}
    for value in x:
        if value in frequency:
            frequency[value] += 1
        else:
            frequency[value] = 1

    max_count = 0
    for count in frequency.values():
        if count > max_count:
            max_count = count

    mode = 0
    for value, count in frequency.items():
        if count == max_count:
            mode = value
    return mode


def compare_statistics(manual_stats, lib_stats):
    print("СРАВНЕНИЕ СТАТИСТИЧЕСКИХ ХАРАКТЕРИСТИК")
    print("=" * 70)
    print(f"{'Параметр':<25} {'Ручной расчет':<15} {'Библиотека':<15} {'Разница':<10}")
    print("-" * 70)

    for key in manual_stats.keys():
        manual_val = manual_stats[key]
        lib_val = lib_stats[key]
        diff = abs(manual_val - lib_val)
        print(f"{key:<25} {manual_val:15.6f} {lib_val:15.6f} {diff:10.6f}")


def main():
    n = 100
    n2 = n * 60
    a = 1
    sigma = 3
    x = np.random.normal(a, sigma, n)
    x2 = np.random.normal(a, sigma, n2)

    count_bins, bound_bins = absolute_frequencies(x)
    rel_count_bins = relative_frequencies(count_bins, n)
    plot_many_hist(x)
    plot_abs_hist(x)
    plot_rel_hist(x, a, sigma)
    plot_cumulate(x, a, sigma)
    plot_boxplot(x)

    manual_statistics = manual_stats(x)
    lib_statistics = library_stats(x)
    compare_statistics(manual_statistics, lib_statistics)

    lib_stats_2 = library_stats(x2)
    print(lib_stats_2)

if __name__ == '__main__':
    main()