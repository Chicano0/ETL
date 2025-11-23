
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("U4: ANÁLISIS NO SUPERVISADO - CLUSTERING DE PRODUCTOS")
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
# PASO 3: INGENIERÍA DE CARACTERÍSTICAS
# =============================================================================
print("\n[3] CREANDO CARACTERÍSTICAS PARA CLUSTERING...")

# Crear características agregadas por producto
df_clustering = df.groupby('producto').agg({
    'precio': ['mean', 'std', 'min', 'max'],
    'cantidad': ['sum', 'mean', 'count'],
}).reset_index()

df_clustering.columns = ['producto', 'precio_promedio', 'precio_std', 'precio_min', 
                         'precio_max', 'cantidad_total', 'cantidad_promedio', 'num_ventas']

# Rellenar NaN en desviación estándar (ocurre cuando hay solo 1 venta)
df_clustering['precio_std'] = df_clustering['precio_std'].fillna(0)

# Crear características adicionales
df_clustering['rango_precio'] = df_clustering['precio_max'] - df_clustering['precio_min']
df_clustering['valor_total_ventas'] = df_clustering['precio_promedio'] * df_clustering['cantidad_total']

print(f"  ✓ Productos únicos: {len(df_clustering)}")
print(f"  ✓ Características creadas: {list(df_clustering.columns[1:])}")

# =============================================================================
# PASO 4: PREPARAR DATOS PARA CLUSTERING
# =============================================================================
print("\n[4] PREPARANDO DATOS...")

# Seleccionar características numéricas
features = ['precio_promedio', 'cantidad_total', 'num_ventas', 
            'rango_precio', 'valor_total_ventas']

X = df_clustering[features].values

# Escalado (CRUCIAL para clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"  ✓ Forma de datos: {X_scaled.shape}")
print(f"  ✓ Features utilizados: {features}")

# =============================================================================
# PASO 5: DETERMINAR NÚMERO ÓPTIMO DE CLUSTERS (MÉTODO ELBOW)
# =============================================================================
print("\n[5] DETERMINANDO NÚMERO ÓPTIMO DE CLUSTERS...")

inertias = []
silhouette_scores = []
k_range = range(2, min(8, len(X_scaled)))

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Gráfica del método Elbow
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Número de Clusters (k)')
axes[0].set_ylabel('Inercia (Within-Cluster Sum of Squares)')
axes[0].set_title('Método del Codo (Elbow Method)', fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Número de Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Análisis de Silhouette', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_elbow_silhouette.png', dpi=300, bbox_inches='tight')
print("  ✓ 01_elbow_silhouette.png")
plt.close()

# Seleccionar k óptimo (el que maximiza silhouette)
k_optimo = k_range[np.argmax(silhouette_scores)]
print(f"  ✓ K óptimo sugerido: {k_optimo} (Silhouette: {max(silhouette_scores):.4f})")

# =============================================================================
# PASO 6: MODELO 1 - K-MEANS INICIAL
# =============================================================================
print("\n[6] APLICANDO K-MEANS INICIAL...")

k_inicial = 3  # Empezamos con 3 clusters
kmeans_inicial = KMeans(n_clusters=k_inicial, random_state=42, n_init=10)
clusters_kmeans_inicial = kmeans_inicial.fit_predict(X_scaled)

# Métricas
silhouette_kmeans_inicial = silhouette_score(X_scaled, clusters_kmeans_inicial)
calinski_kmeans_inicial = calinski_harabasz_score(X_scaled, clusters_kmeans_inicial)
davies_kmeans_inicial = davies_bouldin_score(X_scaled, clusters_kmeans_inicial)

print(f"  K-Means con k={k_inicial}:")
print(f"  ✓ Silhouette Score:      {silhouette_kmeans_inicial:.4f}")
print(f"  ✓ Calinski-Harabasz:     {calinski_kmeans_inicial:.4f}")
print(f"  ✓ Davies-Bouldin:        {davies_kmeans_inicial:.4f}")

# =============================================================================
# PASO 7: MODELO 2 - K-MEANS OPTIMIZADO
# =============================================================================
print("\n[7] APLICANDO K-MEANS OPTIMIZADO...")

kmeans_optimizado = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
clusters_kmeans_opt = kmeans_optimizado.fit_predict(X_scaled)

silhouette_kmeans_opt = silhouette_score(X_scaled, clusters_kmeans_opt)
calinski_kmeans_opt = calinski_harabasz_score(X_scaled, clusters_kmeans_opt)
davies_kmeans_opt = davies_bouldin_score(X_scaled, clusters_kmeans_opt)

print(f"  K-Means con k={k_optimo}:")
print(f"  ✓ Silhouette Score:      {silhouette_kmeans_opt:.4f}")
print(f"  ✓ Calinski-Harabasz:     {calinski_kmeans_opt:.4f}")
print(f"  ✓ Davies-Bouldin:        {davies_kmeans_opt:.4f}")

# =============================================================================
# PASO 8: MODELO 3 - CLUSTERING JERÁRQUICO
# =============================================================================
print("\n[8] APLICANDO CLUSTERING JERÁRQUICO...")

hierarchical = AgglomerativeClustering(n_clusters=k_optimo, linkage='ward')
clusters_hierarchical = hierarchical.fit_predict(X_scaled)

silhouette_hier = silhouette_score(X_scaled, clusters_hierarchical)
calinski_hier = calinski_harabasz_score(X_scaled, clusters_hierarchical)
davies_hier = davies_bouldin_score(X_scaled, clusters_hierarchical)

print(f"  Clustering Jerárquico con k={k_optimo}:")
print(f"  ✓ Silhouette Score:      {silhouette_hier:.4f}")
print(f"  ✓ Calinski-Harabasz:     {calinski_hier:.4f}")
print(f"  ✓ Davies-Bouldin:        {davies_hier:.4f}")

# =============================================================================
# PASO 9: MODELO 4 - DBSCAN
# =============================================================================
print("\n[9] APLICANDO DBSCAN...")

# Parámetros ajustados para datasets pequeños
dbscan = DBSCAN(eps=0.5, min_samples=2)
clusters_dbscan = dbscan.fit_predict(X_scaled)

n_clusters_dbscan = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
n_noise = list(clusters_dbscan).count(-1)

print(f"  DBSCAN (eps=0.5, min_samples=2):")
print(f"  ✓ Clusters encontrados:  {n_clusters_dbscan}")
print(f"  ✓ Puntos de ruido:       {n_noise}")

if n_clusters_dbscan > 1:
    # Filtrar ruido para calcular métricas
    mask = clusters_dbscan != -1
    if mask.sum() > 1:
        silhouette_dbscan = silhouette_score(X_scaled[mask], clusters_dbscan[mask])
        calinski_dbscan = calinski_harabasz_score(X_scaled[mask], clusters_dbscan[mask])
        davies_dbscan = davies_bouldin_score(X_scaled[mask], clusters_dbscan[mask])
        print(f"  ✓ Silhouette Score:      {silhouette_dbscan:.4f}")
        print(f"  ✓ Calinski-Harabasz:     {calinski_dbscan:.4f}")
        print(f"  ✓ Davies-Bouldin:        {davies_dbscan:.4f}")
    else:
        silhouette_dbscan = calinski_dbscan = davies_dbscan = 0
        print("  ⚠ No hay suficientes clusters para métricas")
else:
    silhouette_dbscan = calinski_dbscan = davies_dbscan = 0
    print("  ⚠ DBSCAN no encontró clusters válidos")

# =============================================================================
# PASO 10: REDUCCIÓN DE DIMENSIONALIDAD CON PCA
# =============================================================================
print("\n[10] APLICANDO PCA (REDUCCIÓN DE DIMENSIONALIDAD)...")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

varianza_explicada = pca.explained_variance_ratio_
print(f"  ✓ Varianza explicada por PC1: {varianza_explicada[0]*100:.2f}%")
print(f"  ✓ Varianza explicada por PC2: {varianza_explicada[1]*100:.2f}%")
print(f"  ✓ Varianza total explicada:   {sum(varianza_explicada)*100:.2f}%")

# =============================================================================
# PASO 11: TABLA COMPARATIVA DE MODELOS
# =============================================================================
print("\n[11] TABLA COMPARATIVA DE MODELOS")
print("="*80)

resultados = pd.DataFrame({
    'Modelo': [
        f'K-Means (k={k_inicial})', 
        f'K-Means Opt (k={k_optimo})',
        f'Jerárquico (k={k_optimo})',
        'DBSCAN'
    ],
    'Silhouette': [
        silhouette_kmeans_inicial,
        silhouette_kmeans_opt,
        silhouette_hier,
        silhouette_dbscan
    ],
    'Calinski-Harabasz': [
        calinski_kmeans_inicial,
        calinski_kmeans_opt,
        calinski_hier,
        calinski_dbscan
    ],
    'Davies-Bouldin': [
        davies_kmeans_inicial,
        davies_kmeans_opt,
        davies_hier,
        davies_dbscan
    ]
})

print("\n" + resultados.to_string(index=False))
resultados.to_csv('resultados_U4.csv', index=False)

# =============================================================================
# PASO 12: VISUALIZACIÓN DE CLUSTERS CON PCA
# =============================================================================
print("\n[12] GENERANDO VISUALIZACIONES...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. K-Means Inicial
scatter1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_kmeans_inicial, 
                              cmap='viridis', s=100, alpha=0.7, edgecolors='black')
axes[0, 0].set_title(f'K-Means Inicial (k={k_inicial})', fontweight='bold', fontsize=12)
axes[0, 0].set_xlabel(f'PC1 ({varianza_explicada[0]*100:.1f}%)')
axes[0, 0].set_ylabel(f'PC2 ({varianza_explicada[1]*100:.1f}%)')
axes[0, 0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0, 0], label='Cluster')

# 2. K-Means Optimizado
scatter2 = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_kmeans_opt, 
                              cmap='viridis', s=100, alpha=0.7, edgecolors='black')
axes[0, 1].set_title(f'K-Means Optimizado (k={k_optimo})', fontweight='bold', fontsize=12)
axes[0, 1].set_xlabel(f'PC1 ({varianza_explicada[0]*100:.1f}%)')
axes[0, 1].set_ylabel(f'PC2 ({varianza_explicada[1]*100:.1f}%)')
axes[0, 1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[0, 1], label='Cluster')

# 3. Clustering Jerárquico
scatter3 = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_hierarchical, 
                              cmap='viridis', s=100, alpha=0.7, edgecolors='black')
axes[1, 0].set_title(f'Clustering Jerárquico (k={k_optimo})', fontweight='bold', fontsize=12)
axes[1, 0].set_xlabel(f'PC1 ({varianza_explicada[0]*100:.1f}%)')
axes[1, 0].set_ylabel(f'PC2 ({varianza_explicada[1]*100:.1f}%)')
axes[1, 0].grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=axes[1, 0], label='Cluster')

# 4. DBSCAN
scatter4 = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_dbscan, 
                              cmap='viridis', s=100, alpha=0.7, edgecolors='black')
axes[1, 1].set_title('DBSCAN', fontweight='bold', fontsize=12)
axes[1, 1].set_xlabel(f'PC1 ({varianza_explicada[0]*100:.1f}%)')
axes[1, 1].set_ylabel(f'PC2 ({varianza_explicada[1]*100:.1f}%)')
axes[1, 1].grid(True, alpha=0.3)
plt.colorbar(scatter4, ax=axes[1, 1], label='Cluster')

plt.tight_layout()
plt.savefig('02_visualizacion_clusters_pca.png', dpi=300, bbox_inches='tight')
print("  ✓ 02_visualizacion_clusters_pca.png")
plt.close()

# =============================================================================
# PASO 13: DENDROGRAMA (CLUSTERING JERÁRQUICO)
# =============================================================================
print("\n[13] GENERANDO DENDROGRAMA...")

plt.figure(figsize=(12, 6))
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix, labels=df_clustering['producto'].values, 
           leaf_font_size=8, leaf_rotation=90)
plt.title('Dendrograma - Clustering Jerárquico', fontweight='bold', fontsize=14)
plt.xlabel('Productos')
plt.ylabel('Distancia')
plt.tight_layout()
plt.savefig('03_dendrograma.png', dpi=300, bbox_inches='tight')
print("  ✓ 03_dendrograma.png")
plt.close()

# =============================================================================
# PASO 14: COMPARACIÓN DE MÉTRICAS
# =============================================================================
print("\n[14] GENERANDO GRÁFICAS COMPARATIVAS...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metricas = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin']
datos_metricas = [
    resultados['Silhouette'].values,
    resultados['Calinski-Harabasz'].values,
    resultados['Davies-Bouldin'].values
]

for idx, (ax, metrica, datos) in enumerate(zip(axes, metricas, datos_metricas)):
    x_pos = np.arange(len(resultados))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ax.bar(x_pos, datos, color=colors)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title(metrica, fontweight='bold', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['K-Means\nInicial', 'K-Means\nOpt', 'Jerárquico', 'DBSCAN'], 
                       fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(datos):
        if v > 0:
            ax.text(i, v + max(datos)*0.02, f'{v:.2f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('04_comparacion_metricas.png', dpi=300, bbox_inches='tight')
print("  ✓ 04_comparacion_metricas.png")
plt.close()

# =============================================================================
# PASO 15: PERFIL DE CLUSTERS (MEJOR MODELO)
# =============================================================================
print("\n[15] GENERANDO PERFIL DE CLUSTERS...")

# Usar el modelo con mejor Silhouette
mejor_modelo_idx = resultados['Silhouette'].idxmax()
mejor_modelo_nombre = resultados.loc[mejor_modelo_idx, 'Modelo']

if 'K-Means Opt' in mejor_modelo_nombre:
    clusters_final = clusters_kmeans_opt
elif 'Jerárquico' in mejor_modelo_nombre:
    clusters_final = clusters_hierarchical
elif 'DBSCAN' in mejor_modelo_nombre and n_clusters_dbscan > 0:
    clusters_final = clusters_dbscan
else:
    clusters_final = clusters_kmeans_opt

df_clustering['cluster'] = clusters_final

# Guardar resultados con clusters
df_clustering.to_csv('productos_clustered_U4.csv', index=False)

# Perfil de cada cluster
print(f"\n  Perfil de Clusters ({mejor_modelo_nombre}):")
print("  " + "="*76)

for cluster_id in sorted(df_clustering['cluster'].unique()):
    if cluster_id == -1:
        print(f"\n  RUIDO (Cluster {cluster_id}):")
    else:
        print(f"\n  CLUSTER {cluster_id}:")
    
    cluster_data = df_clustering[df_clustering['cluster'] == cluster_id]
    print(f"    • Productos: {len(cluster_data)}")
    print(f"    • Precio promedio: ${cluster_data['precio_promedio'].mean():.2f}")
    print(f"    • Cantidad total vendida: {cluster_data['cantidad_total'].sum():.0f}")
    print(f"    • Número de ventas: {cluster_data['num_ventas'].sum():.0f}")
    print(f"    • Productos: {', '.join(cluster_data['producto'].values[:5])}")

# =============================================================================
# PASO 16: VARIANZA EXPLICADA POR PCA
# =============================================================================
print("\n[16] GENERANDO GRÁFICA DE VARIANZA EXPLICADA...")

pca_full = PCA()
pca_full.fit(X_scaled)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1), 
        pca_full.explained_variance_ratio_, color='steelblue')
plt.xlabel('Componente Principal')
plt.ylabel('Varianza Explicada')
plt.title('Varianza Explicada por Componente', fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
         np.cumsum(pca_full.explained_variance_ratio_), 'o-', linewidth=2, markersize=8)
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada')
plt.title('Varianza Acumulada', fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_varianza_pca.png', dpi=300, bbox_inches='tight')
print("  ✓ 05_varianza_pca.png")
plt.close()

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "="*80)
print("✅ ANÁLISIS NO SUPERVISADO COMPLETADO EXITOSAMENTE")
print("="*80)

print(f"\n📊 ARCHIVOS GENERADOS:")
print(f"  ✓ reporte_mensual.csv")
print(f"  ✓ resultados_U4.csv")
print(f"  ✓ productos_clustered_U4.csv")
print(f"  ✓ 01_elbow_silhouette.png")
print(f"  ✓ 02_visualizacion_clusters_pca.png")
print(f"  ✓ 03_dendrograma.png")
print(f"  ✓ 04_comparacion_metricas.png")
print(f"  ✓ 05_varianza_pca.png")

print(f"\n🏆 MEJOR MODELO: {mejor_modelo_nombre}")
print(f"  • Silhouette Score:     {resultados.loc[mejor_modelo_idx, 'Silhouette']:.4f}")
print(f"  • Calinski-Harabasz:    {resultados.loc[mejor_modelo_idx, 'Calinski-Harabasz']:.4f}")
print(f"  • Davies-Bouldin:       {resultados.loc[mejor_modelo_idx, 'Davies-Bouldin']:.4f}")

print(f"\n📈 REDUCCIÓN DE DIMENSIONALIDAD (PCA):")
print(f"  • Varianza explicada (2 componentes): {sum(varianza_explicada)*100:.2f}%")
print(f"  • Componentes principales óptimos: 2")

print("\n" + "="*80)