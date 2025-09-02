import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import utils  # Importar funciones de utilidad existentes

# ==================== PARÁMETROS DE CONFIGURACIÓN ====================
# Ajustar estos valores para optimizar la detección según la imagen

# Parámetros de segmentación inicial
LAMBDA_PROJECTION = 0.7  # Peso del canal b en proyección P = a + λ*b (0.4-0.9)
B_PERCENTILE_MIN = 35    # Percentil mínimo para filtro en canal b (20-50)

# Parámetros de limpieza morfológica
CLOSE_KERNEL_SIZE = 7    # Kernel para operación de cierre (5-9)
MEDIAN_BLUR_SIZE = 3     # Tamaño del filtro mediano (3-7)

# Parámetros de separación por watershed
ALPHA_THRESHOLD = 0.4    # Umbral relativo para seeds (0.3-0.6)
DILATE_KERNEL_SIZE = 5   # Kernel para dilatación de sure_bg (3-7)
ERODE_ITERATIONS = 1     # Iteraciones de erosión para seeds

# Parámetros de conteo
AREA_MIN_FACTOR = 0.35   # Factor mínimo de área respecto a mediana (0.2-0.5)
AREA_MAX_FACTOR = 1.75   # Factor máximo para área individual (1.5-2.0)
RADIUS_ESTIMATE = 15     # Radio estimado de RBC para detección de picos (10-20)
PEAK_HEIGHT = 2.0        # Altura mínima de picos locales (1.0-3.0)

# ==================== FUNCIONES DE DETECCIÓN DE CENTROIDES ====================

def analyze_connected_components(mask):
    """
    Analiza componentes conectados y calcula estadísticas de área.
    
    Args:
        mask: Máscara binaria de entrada
        
    Returns:
        tuple: (num_components, labels, stats, centroids, areas, area_stats)
    """
    num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num_components > 1 else np.array([])
    
    area_stats = {}
    if areas.size > 0:
        median_area = float(np.median(areas))
        area_min = max(40, AREA_MIN_FACTOR * median_area)
        area_single_max = AREA_MAX_FACTOR * median_area
        
        area_stats = {
            'median': median_area,
            'min_threshold': area_min,
            'single_max': area_single_max,
            'total_areas': areas
        }
        
        print(f"✓ Análisis de áreas - Mediana: {median_area:.1f}, "
              f"Min: {area_min:.1f}, Max individual: {area_single_max:.1f}")
    
    return num_components, labels, stats, centroids, areas, area_stats

def detect_local_peaks(mask):
    """
    Detecta picos locales usando transformada de distancia para separar RBC agrupados.
    
    Args:
        mask: Máscara binaria de RBC
        
    Returns:
        tuple: (peaks_mask, peak_centroids, num_peaks)
    """
    # Transformada de distancia con suavizado
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_blur = cv2.GaussianBlur(dist_transform, (0, 0), 0.5)
    
    # Configurar kernel para detección de máximos locales
    kernel_size = 2 * RADIUS_ESTIMATE + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Detectar máximos locales
    dilated = cv2.dilate(dist_blur, kernel)
    local_maxima = (dist_blur >= dilated - 1e-6) & (mask > 0)
    
    # Filtrar picos por altura relativa
    mean_filter = cv2.blur(dist_blur, (kernel_size, kernel_size))
    peaks = local_maxima & (dist_blur >= (mean_filter + PEAK_HEIGHT))
    peaks_u8 = (peaks.astype(np.uint8) * 255)
    
    # Obtener centroides de los picos
    num_peaks, peak_labels, peak_stats, peak_centroids = cv2.connectedComponentsWithStats(peaks_u8, 8)
    
    print(f"✓ Detección de picos completada - {num_peaks-1} picos detectados")
    
    return peaks_u8, peak_centroids, num_peaks

def extract_single_cell_centroids(components_data, area_stats):
    """
    Extrae centroides de células individuales (no agrupadas).
    
    Args:
        components_data: Tupla con (num_components, labels, stats, centroids)
        area_stats: Diccionario con estadísticas de área
        
    Returns:
        list: Lista de centroides (x, y) de células individuales
    """
    num_components, labels, stats, centroids = components_data
    single_centroids = []
    
    if area_stats and len(area_stats) > 0:
        area_min = area_stats['min_threshold']
        area_single_max = area_stats['single_max']
        
        for i in range(1, num_components):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Verificar si es una célula individual válida
            if area_min <= area <= area_single_max:
                cx, cy = centroids[i]
                single_centroids.append((int(round(cx)), int(round(cy))))
    
    print(f"✓ Células individuales detectadas: {len(single_centroids)}")
    return single_centroids

def extract_grouped_cell_centroids(components_data, area_stats, peak_data):
    """
    Extrae centroides de células agrupadas usando detección de picos.
    
    Args:
        components_data: Tupla con (num_components, labels, stats, centroids)
        area_stats: Diccionario con estadísticas de área
        peak_data: Tupla con (peaks_mask, peak_centroids, num_peaks)
        
    Returns:
        list: Lista de centroides (x, y) de células en grupos
    """
    num_components, labels, stats, centroids = components_data
    peaks_mask, peak_centroids, num_peaks = peak_data
    grouped_centroids = []
    
    if area_stats and len(area_stats) > 0:
        area_min = area_stats['min_threshold']
        area_single_max = area_stats['single_max']
        
        # Procesar componentes grandes que contienen múltiples células
        for i in range(1, num_components):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Solo procesar componentes que exceden el umbral individual
            if area > area_single_max and area >= area_min:
                # Buscar todos los picos dentro de este componente
                for j in range(1, num_peaks):
                    px, py = peak_centroids[j]
                    py_int, px_int = int(round(py)), int(round(px))
                    
                    # Verificar que el pico esté dentro del componente actual
                    if (0 <= py_int < labels.shape[0] and 
                        0 <= px_int < labels.shape[1] and 
                        labels[py_int, px_int] == i):
                        grouped_centroids.append((px_int, py_int))
    
    print(f"✓ Células agrupadas separadas: {len(grouped_centroids)}")
    return grouped_centroids

def detect_rbc_centroids(final_mask):
    """
    Función principal para detectar centroides de glóbulos rojos.
    Combina análisis de componentes conectados con detección de picos.
    
    Args:
        final_mask: Máscara binaria final de RBC
        
    Returns:
        list: Lista de centroides (x, y) de todos los RBC detectados
    """
    print("🔍 Iniciando detección de centroides...")
    
    # Paso 1: Analizar componentes conectados
    components_data = analyze_connected_components(final_mask)
    num_components, labels, stats, centroids, areas, area_stats = components_data
    
    all_centroids = []
    
    if areas.size == 0:
        print("⚠️  No se encontraron componentes válidos")
        return all_centroids
    
    # Paso 2: Detectar picos locales para separar células agrupadas
    peak_data = detect_local_peaks(final_mask)
    
    # Paso 3: Extraer centroides de células individuales
    single_centroids = extract_single_cell_centroids(
        (num_components, labels, stats, centroids), area_stats
    )
    
    # Paso 4: Extraer centroides de células agrupadas
    grouped_centroids = extract_grouped_cell_centroids(
        (num_components, labels, stats, centroids), area_stats, peak_data
    )
    
    # Paso 5: Combinar todos los centroides
    all_centroids = single_centroids + grouped_centroids
    
    print(f"✅ Detección completada - Total: {len(all_centroids)} RBC "
          f"({len(single_centroids)} individuales + {len(grouped_centroids)} agrupadas)")
    
    return all_centroids

def show_image(img, title, figsize=(6, 5)):
    """Muestra una imagen con título."""
    plt.figure(figsize=figsize)
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_images_grid(images, titles, figsize=(15, 5), cmap='gray'):
    """Muestra múltiples imágenes en una fila."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 3:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            axes[i].imshow(img, cmap=cmap)
        axes[i].set_title(title)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def plot_histogram(img, title, mask=None, bins=256, figsize=(8, 4)):
    """Grafica histograma de intensidades."""
    plt.figure(figsize=figsize)
    if mask is not None:
        data = img[mask == 255]
    else:
        data = img.ravel()
    
    plt.hist(data, bins=bins, range=[0, 256], color='gray', alpha=0.7)
    plt.title(title)
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia")
    plt.grid(True, alpha=0.3)
    plt.show()

def fill_holes(binary_img):
    """Rellena agujeros en una máscara binaria usando flood fill."""
    h, w = binary_img.shape
    flood = binary_img.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(binary_img, inv)
    return filled

# ==================== PIPELINE PRINCIPAL ====================

def detect_red_blood_cells(image_path):
    """
    Pipeline completo para detección y conteo de glóbulos rojos.
    
    Args:
        image_path: Ruta a la imagen de células sanguíneas
        
    Returns:
        tuple: (imagen_original, mascara_final, centroides, conteo)
    """
    
    print("=== INICIANDO PIPELINE DE DETECCIÓN DE GLÓBULOS ROJOS ===\n")
    
    # -------------------- CARGA DE IMAGEN --------------------
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    
    print("✓ Imagen cargada exitosamente")
    plt.figure(figsize=(6, 5))
    utils.imshow(bgr)
    plt.title("Imagen Original")
    plt.show()
    
    # -------------------- CONVERSIÓN A LAB --------------------
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    
    print("✓ Conversión a espacio LAB completada")
    utils.show_images([L, A, B], 
                     titles=["Canal L (Luminancia)", "Canal a (Verde-Rojo)", "Canal b (Azul-Amarillo)"])
    plt.show()
    
    # Histogramas de canales LAB
    plot_histogram(A, "Histograma Canal a (Lab)")
    plot_histogram(B, "Histograma Canal b (Lab)")
    
    # -------------------- SEGMENTACIÓN INICIAL --------------------
    # Separar células del fondo usando Otsu en canal 'a'
    _, cells_mask = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cells_mask = cv2.morphologyEx(cells_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cells_mask = cv2.morphologyEx(cells_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    print("✓ Segmentación inicial (células vs fondo) completada")
    plt.figure(figsize=(6, 5))
    utils.imshow(cells_mask)
    plt.title("Paso 1: Máscara de Células (Otsu en canal 'a')")
    plt.show()
    
    # -------------------- SEGMENTACIÓN DE GLÓBULOS ROJOS --------------------
    # Extraer valores dentro de las células
    vals_a = A[cells_mask == 255].astype(np.float32)
    vals_b = B[cells_mask == 255].astype(np.float32)
    
    if vals_a.size == 0:
        raise ValueError("La máscara de células está vacía. Revisar umbralización.")
    
    # Proyección lineal que favorece glóbulos rojos: P = a + λ*b
    P = (A.astype(np.float32) + LAMBDA_PROJECTION * B.astype(np.float32))
    
    # Normalizar P para Otsu
    P_normalized = np.zeros_like(P, dtype=np.uint8)
    cv2.normalize(P, P_normalized, 0, 255, cv2.NORM_MINMAX)
    P_in_cells = P_normalized[cells_mask == 255]
    
    # Aplicar Otsu en la proyección
    threshold_otsu, _ = cv2.threshold(P_in_cells, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Crear máscara de RBC
    rbc_mask_raw = np.zeros_like(P_normalized, dtype=np.uint8)
    rbc_mask_raw[(P_normalized >= threshold_otsu) & (cells_mask == 255)] = 255
    
    # Filtro adicional en canal 'b' para eliminar células blancas
    B_min = np.percentile(vals_b, B_PERCENTILE_MIN)
    mask_b_filter = cv2.inRange(B, int(B_min), 255)
    rbc_mask_raw = cv2.bitwise_and(rbc_mask_raw, mask_b_filter)
    
    print(f"✓ Segmentación de RBC completada (λ={LAMBDA_PROJECTION}, umbral Otsu={threshold_otsu})")
    
    # Histograma de la proyección
    utils.imshow(P, cmap = 'gray') 
    plt.title(f"Proyección P (λ={LAMBDA_PROJECTION})")
    plt.show()

    utils.imshow(rbc_mask_raw)
    plt.title("RBC Segmentados (Raw)")
    plt.show()
    
    # -------------------- LIMPIEZA MORFOLÓGICA --------------------
    kernel_close = np.ones((CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE), np.uint8)
    rbc_mask = cv2.morphologyEx(rbc_mask_raw, cv2.MORPH_CLOSE, kernel_close)
    rbc_mask = fill_holes(rbc_mask)
    rbc_mask = cv2.medianBlur(rbc_mask, MEDIAN_BLUR_SIZE)
    
    print("✓ Limpieza morfológica completada")
    utils.imshow(rbc_mask)
    plt.title("RBC Limpio (Close + Fill Holes)")
    plt.show()
    
    # -------------------- SEPARACIÓN POR WATERSHED --------------------
    # Transformada de distancia
    dist_transform = cv2.distanceTransform(rbc_mask, cv2.DIST_L2, 5)
    dist_normalized = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # Seeds (foreground seguro)
    _, sure_fg = cv2.threshold(dist_normalized, ALPHA_THRESHOLD, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_ERODE, 
                              np.ones((3, 3), np.uint8), ERODE_ITERATIONS)
    
    # Background seguro
    kernel_dilate = np.ones((DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE), np.uint8)
    sure_bg = cv2.dilate(rbc_mask, kernel_dilate, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    print(f"✓ Preparación para Watershed completada (α={ALPHA_THRESHOLD})")
    utils.show_images([dist_normalized, sure_fg, unknown],
                     titles=["Distancia Transformada", f"Seeds (α={ALPHA_THRESHOLD})", "Región Unknown"],
                     cmap='viridis')
    plt.show()
    
    # Aplicar Watershed
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    markers_watershed = cv2.watershed(bgr.copy(), markers)
    
    # Visualizar resultado del watershed
    ws_overlay = bgr.copy()
    ws_overlay[markers_watershed == -1] = (0, 0, 255)  # Bordes en rojo
    
    print("✓ Watershed aplicado exitosamente")
    plt.figure(figsize=(6, 5))
    utils.imshow(ws_overlay)
    plt.title("Resultado Watershed (bordes en rojo)")
    plt.show()
    
    # -------------------- MÁSCARA FINAL --------------------
    final_mask = np.zeros_like(rbc_mask, dtype=np.uint8)
    final_mask[(markers_watershed > 1) & (rbc_mask > 0)] = 255
    
    plt.figure(figsize=(6, 5))
    utils.imshow(final_mask)
    plt.title("Máscara Final de RBC Separados")
    plt.show()
    
    # -------------------- CONTEO POR CENTROIDES --------------------
    detected_centroids = detect_rbc_centroids(final_mask)
    
    # -------------------- VISUALIZACIÓN FINAL --------------------
    result_image = bgr.copy()
    for (x, y) in detected_centroids:
        cv2.circle(result_image, (x, y), 4, (0, 0, 0), -1)      # Punto negro
        cv2.circle(result_image, (x, y), 7, (255, 255, 255), 1)  # Aro blanco
    
    total_count = len(detected_centroids)
    
    print(f"\n=== RESULTADO FINAL ===")
    print(f"Glóbulos rojos detectados: {total_count}")
    
    utils.imshow(result_image)
    plt.title(f"Resultado (RBC = {total_count})")
    plt.show()
    
    return bgr, final_mask, detected_centroids, total_count

# ==================== EJECUCIÓN ====================

if __name__ == "__main__":
    # Ruta de la imagen
    image_path = Path('imgs/inputs/blood_cells.jpg')
    
    try:
        # Ejecutar pipeline
        original, mask, centroids, count = detect_red_blood_cells(image_path)
        
        print(f"\n¡Pipeline completado exitosamente!")
        print(f"Total de glóbulos rojos detectados: {count}")
        
    except Exception as e:
        print(f"Error en el pipeline: {e}")