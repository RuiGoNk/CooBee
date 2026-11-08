import matplotlib.pyplot as plt

# Данные
configs = ['B1\n(Gola LLM)', 'H1\n(Constraint)', 'H2\n(Few-Shot)', 'H3\n(Pro Model)', 'H4\n(Math Rerank)', 'H5\n(Harness)', 'B0\n(Rules)']
ndcg = [0.4871, 0.5167, 0.5994, 0.6281, 0.6456, 0.7282, 0.6048]

# Настраиваем стиль (чтобы выглядело как в статьях)
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-darkgrid')

# Рисуем линию (Ablation Study)
plt.plot(configs[:6], ndcg[:6], marker='o', linestyle='-', color='b', linewidth=2, markersize=8, label='Ablation Study (H1-H5)')

# Добавляем точку B0 и B1 для сравнения
plt.scatter(configs[0], ndcg[0], color='red', s=100, zorder=5, label='B1: Naive LLM')
plt.scatter(configs[6], ndcg[6], color='green', s=100, zorder=5, label='B0: Rules (MVP)')
plt.scatter(configs[5], ndcg[5], color='blue', s=100, zorder=5)

# Подписи
for i, txt in enumerate(ndcg):
    plt.annotate(f"{txt:.4f}", (configs[i], ndcg[i]), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

plt.title('Ablation Study: NDCG@5 by Configuration', fontsize=14, fontweight='bold')
plt.ylabel('NDCG@5', fontsize=12)
plt.ylim(0.40, 0.80) # Чтобы график не был плоским
plt.legend()
plt.tight_layout()

# Сохраняем
plt.savefig('ndcg_results_chart.png', dpi=300)
print("График сохранен в файл ndcg_results_chart.png")