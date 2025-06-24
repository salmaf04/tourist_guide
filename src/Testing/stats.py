import os
import json
import numpy as np
from scipy import stats
from pathlib import Path
from tqdm import tqdm  # Para la barra de progreso (opcional)
import matplotlib.pyplot as plt

def stat_maker():
    # Configuración inicial
    carpeta_json= Path("src/Testing/tests")
    nivel_confianza = 0.95
    alpha = 1 - nivel_confianza # Nivel de significancia (0.05)
    B = 10000 # Número de muestras bootstrap
    # Almacenar todas las puntuaciones
    puntuaciones_M = []
    puntuaciones_S = []
    # Leer todos los archivos JSON en la carpeta
    for nombre_archivo in os.listdir(carpeta_json):
        if nombre_archivo.endswith(".json"):
            ruta_archivo = os.path.join(carpeta_json, nombre_archivo)

            with open(ruta_archivo, 'r') as f:
                data = json.load(f)

                # Extraer resultados del test
                if "resultados" in data:
                    resultados = data["resultados"]
                    puntuaciones_M.append(resultados["M"])
                    puntuaciones_S.append(resultados["S"])

    # Convertir a arrays de numpy
    M = np.array(puntuaciones_M)
    S = np.array(puntuaciones_S)
    n = len(M)
    print(f"\nNúmero de tests: {n}")

    # Calcular diferencia observada
    diferencia_observada = np.mean(S - M)
    print(f"Diferencia observada (S-M): {diferencia_observada:.6f}")

    # Método Bootstrap para estimar la diferencia
    np.random.seed(42)  # Para reproducibilidad
    bootstrap_diferencias = []

    # Generar muestras bootstrap
    for _ in tqdm(range(B), desc="Calculando Bootstrap"):
        # Seleccionar índices aleatorios con reemplazo
        idx = np.random.choice(n, size=n, replace=True)

        # Calcular diferencia de medias para esta muestra
        diff_mean = np.mean(S[idx] - M[idx])
        bootstrap_diferencias.append(diff_mean)

    # Convertir a array
    bootstrap_diferencias = np.array(bootstrap_diferencias)

    # Calcular intervalo de confianza
    percentil_inferior = np.percentile(bootstrap_diferencias, 100 * alpha)
    percentil_superior = np.percentile(bootstrap_diferencias, 100 * (1 - alpha/2))

    # Calcular p-valor bootstrap
    p_valor = np.mean(bootstrap_diferencias <= 0)

    # Determinar si se rechaza H0
    rechazo_H0 = p_valor < alpha
    decision = "Simulación es mejor que Metaheurística" if rechazo_H0 else "No hay evidencia suficiente para descartar H0"

    # Resultados
    print("\n" + "="*60)
    print(f"H0: Simulación no es mejor que Metaheurística")
    print("-"*60)
    print(f"Media modelo Metaheurística: {np.mean(M):.6f}")
    print(f"Media modelo Simulación: {np.mean(S):.6f}")
    print(f"Diferencia observada (Simulación-Metaheurística): {diferencia_observada:.6f}")
    print("-"*60)
    print(f"Intervalo de confianza bootstrap ({nivel_confianza*100:.0f}%):")
    print(f"Límite inferior: {percentil_inferior:.6f}")
    print(f"Límite superior: {percentil_superior:.6f}")
    print("-"*60)
    print(f"p-valor bootstrap: {p_valor:.6f}")
    print(f"Conclusión (95% confianza): {decision}")
    print("="*60)

    # Visualización de resultados
    plt.figure(figsize=(10, 6))
    plt.hist(bootstrap_diferencias, bins=50, alpha=0.7, color='skyblue')
    plt.axvline(x=diferencia_observada, color='r', linestyle='--', label='Diferencia Observada')
    plt.axvline(x=0, color='k', linestyle='-', label='H0: Sin diferencia')
    plt.axvline(x=percentil_inferior, color='g', linestyle=':', label='Límite Confianza')
    plt.axvline(x=percentil_superior, color='g', linestyle=':')
    plt.title(f'Distribución Bootstrap de la Diferencia (S-M)\nMuestras: {B}')
    plt.xlabel('Diferencia de Medias (S-M)')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # Análisis de potencia si es significativo
    if rechazo_H0:
        # Calcular tamaño del efecto (Cohen's d)
        pooled_std = np.sqrt((np.var(M, ddof=1) + np.var(S, ddof=1)) / 2)
        cohen_d = diferencia_observada / pooled_std

        print("\n" + "="*60)
        print("Análisis Adicional (Diferencia Significativa):")
        print(f"Tamaño del efecto (Cohen's d): {cohen_d:.4f}")

        # Interpretar tamaño del efecto
        if cohen_d < 0.2:
            tamano_efecto = "Muy pequeño"
        elif cohen_d < 0.5:
            tamano_efecto = "Pequeño"
        elif cohen_d < 0.8:
            tamano_efecto = "Mediano"
        else:
            tamano_efecto = "Grande"

        print(f"Magnitud: {tamano_efecto}")
        print("="*60)

if __name__ == "__main__": 
    stat_maker()