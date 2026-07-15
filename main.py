import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# Leitura do arquivo
# ==========================
df = pd.read_csv("ecommerce_preparados.csv")

# ==========================
# Análise inicial
# ==========================
print("Primeiras linhas:")
print(df.head())

print("\nInformações:")
print(df.info())

print("\nEstatísticas:")
print(df.describe())

print("\nValores nulos:")
print(df.isnull().sum())

# ==========================
# Histograma
# ==========================
plt.figure(figsize=(8,5))
plt.hist(df["Preço"], bins=20, edgecolor="black")
plt.title("Distribuição dos Preços")
plt.xlabel("Preço")
plt.ylabel("Quantidade")
plt.show()

# ==========================
# Dispersão
# ==========================
plt.figure(figsize=(8,5))
plt.scatter(df["Preço"], df["Nota"])
plt.title("Preço x Nota")
plt.xlabel("Preço")
plt.ylabel("Nota")
plt.show()

# ==========================
# Heatmap
# ==========================
plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include="number").corr(),
            annot=True,
            cmap="coolwarm")
plt.title("Mapa de Correlação")
plt.show()

# ==========================
# Barras
# ==========================
plt.figure(figsize=(10,5))
df["Marca"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Marcas")
plt.xlabel("Marca")
plt.ylabel("Quantidade")
plt.show()

# ==========================
# Pizza
# ==========================
plt.figure(figsize=(8,8))
df["Material"].value_counts().head(5).plot(
    kind="pie",
    autopct="%1.1f%%"
)
plt.title("Distribuição dos Materiais")
plt.ylabel("")
plt.show()

# ==========================
# Boxplot (Gráfico em U)
# ==========================
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Preço"])
plt.title("Boxplot dos Preços")
plt.xlabel("Preço")
plt.show()

# ==========================
# Regressão
# ==========================
plt.figure(figsize=(8,5))
sns.regplot(
    data=df,
    x="Preço",
    y="Nota"
)
plt.title("Regressão Linear entre Preço e Nota")
plt.xlabel("Preço")
plt.ylabel("Nota")
plt.show()