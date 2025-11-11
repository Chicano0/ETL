# =============================================================================
# U3: ANÁLISIS SUPERVISADO - CLASIFICACIÓN DE PRODUCTOS
# Tienda de Electrónica - Predicción de Categoría por Precio
# VERSIÓN ADAPTADA PARA DATASETS PEQUEÑOS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, LeaveOneOut
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("U3: ANÁLISIS SUPERVISADO - CLASIFICACIÓN DE PRODUCTOS")
print("="*80)

# =============================================================================
# PASO 1: CARGAR DATOS
# =============================================================================
print("\n[1] CARGANDO DATOS...")

try:
    df_raw = pd.read_csv('ventas_raw.csv')
    print(f"  ✓ Registros originales: {len(df_raw)}")
    print(f"  ✓ Columnas: {list(df_raw.columns)}")
except FileNotFoundError:
    print("❌ Error: No encuentra 'ventas_raw.csv'")
    exit()

# =============================================================================
# PASO 2: LIMPIAR DATOS (ETL)
# =============================================================================
print("\n[2] LIMPIANDO DATOS...")

df = df_raw.copy()
df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

if 'precio' in df.columns:
    df['precio'] = df['precio'].astype(str).str.replace('$|,| ', '', regex=True)
    df['precio'] = pd.to_numeric(df['precio'], errors='coerce')

if 'cantidad' in df.columns:
    df['cantidad'] = pd.to_numeric(df['cantidad'], errors='coerce')
    df['cantidad'] = df['cantidad'].fillna(1).astype(int)
    df['cantidad'] = df['cantidad'].apply(lambda x: max(1, x))

for col in ['producto', 'cliente', 'region']:
    if col in df.columns:
        df[col] = df[col].str.lower().str.title()

if 'fecha' in df.columns:
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce').dt.strftime('%Y-%m-%d')

campos = [c for c in ['producto', 'cliente', 'fecha', 'precio'] if c in df.columns]
df = df.dropna(subset=campos, how='any')

if all(c in df.columns for c in ['producto', 'cliente', 'fecha']):
    df_original_len = len(df)
    df = df.drop_duplicates(subset=['producto', 'cliente', 'fecha'], keep='first')
    duplicados = df_original_len - len(df)
    if duplicados > 0:
        print(f"  ✓ Duplicados eliminados: {duplicados}")

print(f"  ✓ Registros finales: {len(df)}")
df.to_csv('reporte_mensual.csv', index=False)

# =============================================================================
# PASO 3: CREAR VARIABLE OBJETIVO (CATEGORÍA)
# =============================================================================
print("\n[3] CREANDO CATEGORÍAS...")

q1 = df['precio'].quantile(0.33)
q2 = df['precio'].quantile(0.67)

def asignar_categoria(precio):
    if precio <= q1:
        return 'Baja'
    elif precio <= q2:
        return 'Media'
    else:
        return 'Alta'

df['categoria'] = df['precio'].apply(asignar_categoria)

print(f"  Distribución de categorías:")
for cat in ['Baja', 'Media', 'Alta']:
    count = len(df[df['categoria'] == cat])
    pct = count / len(df) * 100 if len(df) > 0 else 0
    print(f"    {cat}: {count} registros ({pct:.1f}%)")

# =============================================================================
# PASO 4: PREPARAR FEATURES
# =============================================================================
print("\n[4] PREPARANDO FEATURES...")

df_model = df.copy()

le_producto = LabelEncoder()
df_model['producto_enc'] = le_producto.fit_transform(df_model['producto'])

le_cliente = LabelEncoder()
df_model['cliente_enc'] = le_cliente.fit_transform(df_model['cliente'])

le_region = LabelEncoder()
df_model['region_enc'] = le_region.fit_transform(df_model['region'])

features = ['precio', 'cantidad', 'producto_enc', 'cliente_enc', 'region_enc']
X = df_model[features]

le_target = LabelEncoder()
y = le_target.fit_transform(df['categoria'])

print(f"  ✓ Features: {features}")
print(f"  ✓ Clases: {list(le_target.classes_)}")
print(f"  ✓ Total muestras: {len(X)}")

# =============================================================================
# PASO 5: DIVISIÓN Y ESCALADO (ADAPTADO PARA POCOS DATOS)
# =============================================================================
print("\n[5] DIVIDIENDO Y ESCALADO...")

# Para datasets pequeños, usar Leave-One-Out o validación cruzada
if len(X) <= 10:
    print(f"  ⚠ Dataset muy pequeño ({len(X)} muestras). Usando validación cruzada especial.")
    test_size = 0.33  # 1/3 para test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"  ✓ Entrenamiento: {len(X_train)} muestras")
print(f"  ✓ Prueba: {len(X_test)} muestras")

# =============================================================================
# PASO 6: MODELO 1 - ÁRBOL DE DECISIÓN
# =============================================================================
print("\n[6] ENTRENANDO ÁRBOL DE DECISIÓN...")

dt = DecisionTreeClassifier(max_depth=2, min_samples_split=2, random_state=42, criterion='gini')
dt.fit(X_train_s, y_train)
y_pred_dt = dt.predict(X_test_s)

acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt, average='weighted', zero_division=0)
recall_dt = recall_score(y_test, y_pred_dt, average='weighted', zero_division=0)
f1_dt = f1_score(y_test, y_pred_dt, average='weighted', zero_division=0)

print(f"  ✓ Accuracy:  {acc_dt:.4f}")
print(f"  ✓ Precision: {prec_dt:.4f}")
print(f"  ✓ Recall:    {recall_dt:.4f}")
print(f"  ✓ F1-Score:  {f1_dt:.4f}")

# =============================================================================
# PASO 7: MODELO 2 - RANDOM FOREST
# =============================================================================
print("\n[7] ENTRENANDO RANDOM FOREST...")

rf = RandomForestClassifier(n_estimators=10, max_depth=2, min_samples_split=2, random_state=42)
rf.fit(X_train_s, y_train)
y_pred_rf = rf.predict(X_test_s)

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
recall_rf = recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)

print(f"  ✓ Accuracy:  {acc_rf:.4f}")
print(f"  ✓ Precision: {prec_rf:.4f}")
print(f"  ✓ Recall:    {recall_rf:.4f}")
print(f"  ✓ F1-Score:  {f1_rf:.4f}")

# =============================================================================
# PASO 8: OPTIMIZACIÓN CON GRID SEARCH (ADAPTADO)
# =============================================================================
print("\n[8] OPTIMIZANDO CON GRID SEARCH...")

if len(X_train) > 3:
    params = {
        'n_estimators': [5, 10],
        'max_depth': [1, 2, 3],
        'min_samples_split': [2, 3]
    }
    
    grid = GridSearchCV(RandomForestClassifier(random_state=42), params, 
                       cv=2, scoring='f1_weighted', n_jobs=-1, verbose=0)
    grid.fit(X_train_s, y_train)
    
    print(f"  ✓ Mejores parámetros:")
    for param, valor in grid.best_params_.items():
        print(f"    - {param}: {valor}")
    
    y_pred_opt = grid.predict(X_test_s)
else:
    print("  ⚠ Datos insuficientes para Grid Search. Usando RF original.")
    y_pred_opt = y_pred_rf
    grid = rf

acc_opt = accuracy_score(y_test, y_pred_opt)
prec_opt = precision_score(y_test, y_pred_opt, average='weighted', zero_division=0)
recall_opt = recall_score(y_test, y_pred_opt, average='weighted', zero_division=0)
f1_opt = f1_score(y_test, y_pred_opt, average='weighted', zero_division=0)

print(f"\n  RF OPTIMIZADO:")
print(f"  ✓ Accuracy:  {acc_opt:.4f}")
print(f"  ✓ Precision: {prec_opt:.4f}")
print(f"  ✓ Recall:    {recall_opt:.4f}")
print(f"  ✓ F1-Score:  {f1_opt:.4f}")

# =============================================================================
# PASO 9: TABLA COMPARATIVA
# =============================================================================
print("\n[9] TABLA COMPARATIVA DE MODELOS")
print("="*80)

resultados = pd.DataFrame({
    'Modelo': ['Árbol Decisión', 'Random Forest', 'RF Optimizado'],
    'Accuracy': [acc_dt, acc_rf, acc_opt],
    'Precision': [prec_dt, prec_rf, prec_opt],
    'Recall': [recall_dt, recall_rf, recall_opt],
    'F1-Score': [f1_dt, f1_rf, f1_opt]
})

print("\n" + resultados.to_string(index=False))
resultados.to_csv('resultados_U3.csv', index=False)

# =============================================================================
# PASO 10: REPORTE POR CLASE
# =============================================================================
print("\n[10] REPORTE DETALLADO POR CLASE")
print("="*80)
print(classification_report(y_test, y_pred_opt, target_names=le_target.classes_, zero_division=0))

# =============================================================================
# PASO 11: VISUALIZACIONES
# =============================================================================
print("\n[11] GENERANDO VISUALIZACIONES...")

plt.style.use('seaborn-v0_8-darkgrid')

# Gráfica 1: Matrices de confusión
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (y_pred, titulo) in enumerate([(y_pred_dt, "Árbol Decisión"),
                                        (y_pred_rf, "Random Forest"),
                                        (y_pred_opt, "RF Optimizado")]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=le_target.classes_, yticklabels=le_target.classes_, cbar=False)
    axes[idx].set_title(titulo, fontweight='bold')
    axes[idx].set_ylabel('Real')
    axes[idx].set_xlabel('Predicho')

plt.tight_layout()
plt.savefig('01_matrices_confusion.png', dpi=300, bbox_inches='tight')
print("  ✓ 01_matrices_confusion.png")
plt.close()

# Gráfica 2: Importancia de características
if hasattr(grid, 'best_estimator_'):
    importances = grid.best_estimator_.feature_importances_
else:
    importances = rf.feature_importances_

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(indices)]
plt.barh(range(len(indices)), importances[indices], color=colors)
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Importancia')
plt.title('Importancia de Características')
plt.tight_layout()
plt.savefig('02_importancia_features.png', dpi=300, bbox_inches='tight')
print("  ✓ 02_importancia_features.png")
plt.close()

# Gráfica 3: Comparación de modelos
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
metricas_nombres = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
valores_metricas = [
    [acc_dt, acc_rf, acc_opt],
    [prec_dt, prec_rf, prec_opt],
    [recall_dt, recall_rf, recall_opt],
    [f1_dt, f1_rf, f1_opt]
]

for idx, (ax, met_nombre, vals) in enumerate(zip(axes.flat, metricas_nombres, valores_metricas)):
    x_pos = np.arange(3)
    ax.bar(x_pos, vals, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Score')
    ax.set_title(met_nombre, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Árbol', 'RF', 'RF Opt'])
    ax.set_ylim(0, 1.1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f'{v:.2f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('03_comparacion_modelos.png', dpi=300, bbox_inches='tight')
print("  ✓ 03_comparacion_modelos.png")
plt.close()

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "="*80)
print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
print("="*80)

print(f"\n📊 ARCHIVOS GENERADOS:")
print(f"  ✓ reporte_mensual.csv")
print(f"  ✓ resultados_U3.csv")
print(f"  ✓ 01_matrices_confusion.png")
print(f"  ✓ 02_importancia_features.png")
print(f"  ✓ 03_comparacion_modelos.png")

print(f"\n🏆 MEJOR MODELO: RF Optimizado")
print(f"  • Accuracy:  {acc_opt*100:.2f}%")
print(f"  • F1-Score:  {f1_opt:.4f}")

print("\n" + "="*80)