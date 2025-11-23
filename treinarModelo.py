from ultralytics import YOLO
import torch
import psutil
import os

# ============================================================================
# CONFIGURAÇÃO DE HARDWARE: Detecta automaticamente e permite customização manual
# ============================================================================
# 
# EXPLICAÇÃO DE RECURSOS POR CONFIGURAÇÃO:
# 
# ▶ batch_size        → RAM (Memória Principal)
#   Maior = Mais rápido mas usa mais RAM. A cada aumento de 16, use ~500MB-1GB mais
#   
# ▶ imgsz             → VRAM (GPU) / RAM (CPU)
#   Maior = Melhor precisão mas usa mais memória. 640→768→832→1024
#   
# ▶ workers           → CPU (Processador)
#   Maior = Mais núcleos para carregar dados. Use até (núcleos_totais - 2)
#   
# ▶ cache='ram'       → RAM (Memória Principal)
#   Carrega imagens na RAM. Rápido mas precisa de +2-5GB de RAM livre
#   
# ▶ amp=True          → VRAM / RAM (reduz uso em ~50%)
#   Mixed Precision reduz precisão para economizar memória (~50% menos RAM/VRAM)
#
# ============================================================================

# ╔════════════════════════════════════════════════════════════════════════╗
# ║ VALORES MANUAIS - CUSTOMIZE AQUI SE TIVER MAIS RAM/GPU DISPONÍVEL     ║
# ╚════════════════════════════════════════════════════════════════════════╝

# Para MÁXIMO DESEMPENHO com 32GB RAM: use 64. Se amigo tem menos, reduza
CUSTOM_BATCH_SIZE = None  # None = automático | Coloque número (16, 32, 48, 64, 80...) para forçar

# Tamanho da imagem: 640 (padrão), 768 (melhor), 832, ou 1024 (máximo)
# MAIOR = MELHOR PRECISÃO. Com 32GB, use 768 ou 832. CPU pode usar 1024 se tiver tempo
CUSTOM_IMGSZ = 768

# Número de workers: use todos os núcleos que tiver
CUSTOM_WORKERS = None  # None = automático | Coloque 4, 6, 8, 12, 16....

# Cache: 'ram' (mais rápido), 'disk' (economiza RAM), False (sem cache)
# Com 32GB, sempre use 'ram' para máximo de velocidade
CUSTOM_CACHE = 'ram'  # None = automático | Ou force: 'ram', 'disk', False

# ╚════════════════════════════════════════════════════════════════════════╝

def get_device_config():
    """
    Detecta hardware e retorna configuração otimizada.
    Se valores CUSTOM forem definidos, usa aqueles em vez de automático.
    
    Retorna: (device, batch_size, cache_mode, num_workers)
    """
    
    # Detecta GPU disponível
    device = None
    gpu_type = None
    available_vram = 0
    
    if torch.cuda.is_available():
        device = 0
        gpu_type = "CUDA"
        available_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 0
        gpu_type = "MPS (Apple)"
        available_vram = psutil.virtual_memory().total / (1024**3)
    else:
        device = 'cpu'
        gpu_type = "CPU"
    
    # Obtém informações de RAM
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    
    # Obtém número de núcleos CPU
    cpu_count = os.cpu_count() or 4
    
    # ─── BATCH SIZE ───
    if CUSTOM_BATCH_SIZE is not None:
        batch_size = CUSTOM_BATCH_SIZE
    elif available_ram_gb > 24:
        batch_size = 48  # Usa mais RAM disponível
    elif available_ram_gb > 16:
        batch_size = 32
    elif available_ram_gb > 8:
        batch_size = 16
    else:
        batch_size = 8
    
    # ─── WORKERS (carregamento de dados) ───
    if CUSTOM_WORKERS is not None:
        num_workers = CUSTOM_WORKERS
    else:
        num_workers = min(6, cpu_count - 2)  # Deixa 2 núcleos para sistema
    
    # ─── CACHE (onde armazenar imagens) ───
    if CUSTOM_CACHE is not None:
        cache_mode = CUSTOM_CACHE
    elif available_ram_gb > 20:
        cache_mode = 'ram'  # Carrega na RAM - muito rápido
    elif available_ram_gb > 10:
        cache_mode = 'disk'  # Carrega do disco - mais lento que RAM
    else:
        cache_mode = False  # Sem cache - mais lento mas economiza RAM
    
    # ─── EXIBE CONFIGURAÇÃO DETECTADA ───
    print("=" * 75)
    print("CONFIGURAÇÃO DE HARDWARE DETECTADA:")
    print(f"   Dispositivo: {gpu_type}")
    if gpu_type != "CPU":
        print(f"   VRAM GPU: {available_vram:.2f} GB")
    print(f"   RAM Disponível: {available_ram_gb:.2f} GB / {total_ram_gb:.2f} GB")
    print(f"   Núcleos CPU: {cpu_count}")
    print("\nCONFIGURAÇÕES OTIMIZADAS:")
    print(f"   Batch Size: {batch_size} (usa ~{batch_size * 0.3:.1f}MB RAM por ciclo)")
    print(f"   Image Size: {CUSTOM_IMGSZ} pixels (usa ~{(CUSTOM_IMGSZ/640)**2 * 0.5:.1f}GB VRAM/batch)")
    print(f"   Workers: {num_workers} threads (carregadores de dados)")
    print(f"   Cache: {cache_mode if cache_mode else 'Desativado'} (armazenamento de imagens)")
    print(f"   Mixed Precision (AMP): Desativado (ativado, economiza ~50% de memória, mas tem piores resultados)")
    print("=" * 75)
    
    return device, batch_size, cache_mode, num_workers

# Obtém configuração otimizada
device, batch_size, cache_mode, num_workers = get_device_config()

# Carrega modelo YOLOv8 pré-treinado
model = YOLO("yolov8n.pt")  # pequeno, rápido para teste; depois pode usar yolov8s.pt, yolov8m.pt etc.

# Treinar com configurações otimizadas
results = model.train(
    data="D:/TCC/datasets/datasetsNovos/CODEBRIM_split_yolo/data.yaml", #local do YAML do dataset
    epochs=200,  # 20 no teste (aumentar depois para melhor convergência, uns 200 (com early stopping deve parar na casa dos 80))
    imgsz=CUSTOM_IMGSZ,  # Tamanho da imagem (maior = mais preciso)
    batch=batch_size,  # Batch size (maior = mais rápido, usa mais RAM)
    device=device,  # GPU ou CPU
    amp=False,  #  DESATIVADO = máxima precisão em cálculos (melhor qualidade)
    cache=cache_mode,  # Carregar imagens na RAM (muito mais rápido)
    workers=num_workers,  # Workers para carregar dados (maior = mais paralelo)
    patience=10,  # Early stopping se não melhorar em 10 épocas
    name="CODEBRIM_yolo",
    verbose=True #mostra quao completo está o treino
)

# Resultado final
if results:
    print("\nTreinamento completado com sucesso!")

#yolo predict model=best.pt source=imagem.png conf=0.3