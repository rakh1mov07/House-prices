"""
House Prices Analysis
=====================
Предсказание стоимости жилья методами машинного обучения.
"""

# ── 0. Импорты ────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Стиль графиков
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

# ── 1. Загрузка данных ────────────────────────────────────────────────────────
print("=" * 60)
print("  HOUSE PRICES — АНАЛИЗ И ПРЕДСКАЗАНИЕ СТОИМОСТИ ЖИЛЬЯ")
print("=" * 60)

df = pd.read_csv("data/train.csv")
print(f"\n[1] Датасет загружен: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(df.head())

# ── 2. Первичный осмотр ───────────────────────────────────────────────────────
print("\n[2] ПЕРВИЧНЫЙ ОСМОТР")
print("-" * 40)
print(df.dtypes)
print(f"\nПропущенные значения:")
missing = df.isnull().sum()
print(missing[missing > 0])
print(f"\nСтатистика целевой переменной (SalePrice):")
print(df["SalePrice"].describe())

# ── 3. EDA ────────────────────────────────────────────────────────────────────
print("\n[3] РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ (EDA)")

fig = plt.figure(figsize=(16, 14))
fig.suptitle("EDA — House Prices Dataset", fontsize=15, fontweight="bold", y=1.01)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 3.1 Распределение цен
ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(df["SalePrice"], bins=40, kde=True, ax=ax1, color="#2196F3")
ax1.set_title("Распределение SalePrice")
ax1.set_xlabel("Цена ($)")

# 3.2 Log-распределение цен
ax2 = fig.add_subplot(gs[0, 1])
sns.histplot(np.log1p(df["SalePrice"]), bins=40, kde=True, ax=ax2, color="#4CAF50")
ax2.set_title("Лог-распределение SalePrice")
ax2.set_xlabel("log(Цена + 1)")

# 3.3 Цена по качеству
ax3 = fig.add_subplot(gs[0, 2])
sns.boxplot(data=df, x="OverallQual", y="SalePrice", ax=ax3,
            palette="Blues", flierprops={"markersize": 2})
ax3.set_title("Цена vs Качество отделки")
ax3.set_xlabel("OverallQual (1–10)")
ax3.tick_params(axis="x", labelsize=8)

# 3.4 Площадь vs Цена
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(df["GrLivArea"], df["SalePrice"], alpha=0.3, s=12, color="#FF5722")
ax4.set_title("Жилая площадь vs Цена")
ax4.set_xlabel("GrLivArea (кв.фут)")
ax4.set_ylabel("SalePrice ($)")

# 3.5 Год постройки vs Цена
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(df["YearBuilt"], df["SalePrice"], alpha=0.3, s=12, color="#9C27B0")
ax5.set_title("Год постройки vs Цена")
ax5.set_xlabel("YearBuilt")

# 3.6 Корреляционная матрица
ax6 = fig.add_subplot(gs[1:, 2])
corr = df.select_dtypes(include=[np.number]).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax6, linewidths=0.3, annot_kws={"size": 6},
            cbar_kws={"shrink": 0.7})
ax6.set_title("Корреляционная матрица")
ax6.tick_params(labelsize=7)

# 3.7 Гаражи vs Цена
ax7 = fig.add_subplot(gs[2, 0])
valid = df.dropna(subset=["GarageCars"])
sns.boxplot(data=valid, x="GarageCars", y="SalePrice", ax=ax7,
            palette="Set2", flierprops={"markersize": 2})
ax7.set_title("Машиномест в гараже vs Цена")

# 3.8 Площадь подвала vs Цена
ax8 = fig.add_subplot(gs[2, 1])
valid2 = df.dropna(subset=["TotalBsmtSF"])
ax8.scatter(valid2["TotalBsmtSF"], valid2["SalePrice"], alpha=0.3, s=12, color="#00BCD4")
ax8.set_title("Площадь подвала vs Цена")
ax8.set_xlabel("TotalBsmtSF (кв.фут)")

plt.tight_layout()
plt.savefig("eda_plots.png", bbox_inches="tight", dpi=130)
plt.close()
print("  → Сохранено: eda_plots.png")

# Топ корреляций
print("\nТоп корреляций с SalePrice:")
top_corr = corr["SalePrice"].drop("SalePrice").abs().sort_values(ascending=False)
print(top_corr.to_string())

# ── 4. Предобработка ──────────────────────────────────────────────────────────
print("\n[4] ПРЕДОБРАБОТКА ДАННЫХ")

# Целевая переменная — логарифм (нормализуем распределение)
y = np.log1p(df["SalePrice"])
X = df.drop(columns=["SalePrice"])

# Заполнение пропусков
fill_median = ["TotalBsmtSF", "GarageCars", "MasVnrArea"]
for col in fill_median:
    median = X[col].median()
    X[col] = X[col].fillna(median)
    print(f"  {col}: пропуски → медиана ({median:.1f})")

print(f"\nПропусков после обработки: {X.isnull().sum().sum()}")

# Выбросы по GrLivArea (убираем дома > 4000 кв.фут с подозрительно низкой ценой)
outlier_mask = ~((X["GrLivArea"] > 4000) & (y < np.log1p(200000)))
X, y = X[outlier_mask], y[outlier_mask]
print(f"Убрано выбросов: {(~outlier_mask).sum()}")
print(f"Итоговый размер: {X.shape}")

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Масштабирование
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── 5. Обучение моделей ───────────────────────────────────────────────────────
print("\n[5] ОБУЧЕНИЕ МОДЕЛЕЙ")

def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    mae  = mean_absolute_error(y_te, pred)
    r2   = r2_score(y_te, pred)
    # RMSLE в исходных ценах
    rmsle = np.sqrt(mean_squared_error(y_te, pred))  # уже в log-пространстве
    print(f"  {name:<30} RMSE(log)={rmse:.4f}  MAE(log)={mae:.4f}  R²={r2:.4f}")
    return {"model": model, "name": name, "rmse": rmse, "mae": mae, "r2": r2, "pred": pred}

results = []

# Ridge
ridge_params = {"alpha": [0.1, 1, 10, 50, 100, 200]}
ridge_gs = GridSearchCV(Ridge(), ridge_params, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
results.append(evaluate("Ridge (GridSearchCV)", ridge_gs, X_train_sc, X_test_sc, y_train, y_test))
print(f"    best alpha = {ridge_gs.best_params_['alpha']}")

# Lasso
lasso_params = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1]}
lasso_gs = GridSearchCV(Lasso(max_iter=5000), lasso_params, cv=5,
                         scoring="neg_root_mean_squared_error", n_jobs=-1)
results.append(evaluate("Lasso (GridSearchCV)", lasso_gs, X_train_sc, X_test_sc, y_train, y_test))
print(f"    best alpha = {lasso_gs.best_params_['alpha']}")

# Random Forest
rf_params = {"n_estimators": [100, 200], "max_depth": [None, 10, 20],
             "min_samples_split": [2, 5]}
rf_gs = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3,
                     scoring="neg_root_mean_squared_error", n_jobs=-1)
results.append(evaluate("Random Forest (GridSearchCV)", rf_gs, X_train, X_test, y_train, y_test))
print(f"    best params = {rf_gs.best_params_}")

# Gradient Boosting
gb_params = {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
             "max_depth": [3, 5]}
gb_gs = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, cv=3,
                     scoring="neg_root_mean_squared_error", n_jobs=-1)
results.append(evaluate("Gradient Boosting (GridSearchCV)", gb_gs, X_train, X_test, y_train, y_test))
print(f"    best params = {gb_gs.best_params_}")

# ── 6. Cross-validation лучшей модели ────────────────────────────────────────
best = min(results, key=lambda r: r["rmse"])
print(f"\n[6] CROSS-VALIDATION лучшей модели: {best['name']}")

if "Forest" in best["name"] or "Boosting" in best["name"]:
    cv_X, cv_y = X_train, y_train
else:
    cv_X, cv_y = X_train_sc, y_train

cv_scores = cross_val_score(best["model"], cv_X, cv_y,
                             cv=5, scoring="neg_root_mean_squared_error")
cv_rmse = -cv_scores
print(f"  CV RMSE (5-fold): {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

# ── 7. Визуализация результатов ───────────────────────────────────────────────
print("\n[7] ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Результаты моделей машинного обучения", fontsize=14, fontweight="bold")

colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

# 7.1 Сравнение R² моделей
ax = axes[0, 0]
names = [r["name"].split(" (")[0] for r in results]
r2s = [r["r2"] for r in results]
bars = ax.barh(names, r2s, color=colors)
ax.set_xlim(0, 1.05)
ax.set_title("R² по моделям")
ax.set_xlabel("R²")
for bar, val in zip(bars, r2s):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)

# 7.2 RMSE по моделям
ax = axes[0, 1]
rmses = [r["rmse"] for r in results]
bars2 = ax.barh(names, rmses, color=colors)
ax.set_title("RMSE (log) по моделям")
ax.set_xlabel("RMSE")
for bar, val in zip(bars2, rmses):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)

# 7.3 Predicted vs Actual (лучшая модель)
ax = axes[0, 2]
pred_log = best["pred"]
actual_log = y_test.values
ax.scatter(np.expm1(actual_log), np.expm1(pred_log), alpha=0.4, s=15, color="#2196F3")
lims = [min(np.expm1(actual_log)), max(np.expm1(actual_log))]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Идеал")
ax.set_title(f"Predicted vs Actual\n({best['name'].split(' (')[0]})")
ax.set_xlabel("Реальная цена ($)")
ax.set_ylabel("Предсказанная цена ($)")
ax.legend(fontsize=8)

# 7.4 Остатки (residuals)
ax = axes[1, 0]
residuals = pred_log - actual_log
ax.scatter(np.expm1(pred_log), residuals, alpha=0.4, s=15, color="#FF5722")
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.set_title("Остатки (Residuals)")
ax.set_xlabel("Предсказанная цена ($)")
ax.set_ylabel("Остаток (log)")

# 7.5 Feature Importance (если RF или GB)
ax = axes[1, 1]
tree_models = [r for r in results if "Forest" in r["name"] or "Boosting" in r["name"]]
if tree_models:
    best_tree = min(tree_models, key=lambda r: r["rmse"])
    importances = best_tree["model"].best_estimator_.feature_importances_
    feat_names = X.columns.tolist()
    fi_series = pd.Series(importances, index=feat_names).sort_values()
    fi_series.plot(kind="barh", ax=ax, color="#4CAF50")
    ax.set_title(f"Feature Importance\n({best_tree['name'].split(' (')[0]})")
    ax.set_xlabel("Важность")

# 7.6 Распределение остатков
ax = axes[1, 2]
ax.hist(residuals, bins=40, color="#9C27B0", edgecolor="white", alpha=0.8)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_title("Распределение остатков")
ax.set_xlabel("Остаток")
ax.set_ylabel("Частота")

plt.tight_layout()
plt.savefig("model_results.png", bbox_inches="tight", dpi=130)
plt.close()
print("  → Сохранено: model_results.png")

# ── 8. Итоговая таблица ───────────────────────────────────────────────────────
print("\n[8] ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("-" * 60)
print(f"{'Модель':<32} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("-" * 60)
for r in sorted(results, key=lambda x: x["r2"], reverse=True):
    print(f"{r['name'].split(' (')[0]:<32} {r['rmse']:>8.4f} {r['mae']:>8.4f} {r['r2']:>8.4f}")
print("-" * 60)
print(f"\n✓ Лучшая модель: {best['name'].split(' (')[0]}  (R²={best['r2']:.4f})")
print("\nАнализ завершён.")
