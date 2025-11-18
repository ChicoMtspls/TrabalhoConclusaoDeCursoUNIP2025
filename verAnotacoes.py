import cv2
import os
import random
import argparse
import re
import ast
import matplotlib.pyplot as plt
import matplotlib.cm as cm

dataset_path = r"D:/TCC/datasets/datasetsNovos/GYU_DET/train"
#dataset_path = r"D:/TCC/datasets/datasetsNovos/CODEBRIM_original_images/original_dataset/train" #OLD COM PROBLEMA
#dataset_path = r"D:/TCC/datasets/datasetsNovos/CODEBRIM_split_yolo/train" 
img_dir = os.path.join(dataset_path, "images")
lbl_dir = os.path.join(dataset_path, "labels")

# Configuráveis (padrões, podem ser sobrescritos por linha de comando)
num_images = 32          # quantas imagens mostrar
only_annotated = True      # True = preferir apenas imagens com anotações

# --- CLI args ---
parser = argparse.ArgumentParser(description='Visualizar imagens anotadas do dataset')
parser.add_argument('--num', type=int, default=num_images, help='Número de imagens a mostrar')
parser.add_argument('--all', action='store_true', help='Incluir imagens sem anotação (override only_annotated)')
parser.add_argument('--seed', type=int, default=None, help='Semente para randomização (inteiro). Se omitido, usa entropia do sistema')
parser.add_argument('--pattern', type=str, default=None, help='Filtrar nomes de arquivo por substring (case-insensitive)')
args = parser.parse_args()

num_images = args.num
if args.all:
    only_annotated = False

# aplica semente (ou entropia do sistema se None)
random.seed(args.seed)
# Window size controls (pixels). Default 1920x1080 — adjust if it doesn't fit your screen
parser.add_argument('--winw', type=int, default=1920, help='Largura da janela em pixels')
parser.add_argument('--winh', type=int, default=1080, help='Altura da janela em pixels')
args = parser.parse_args()

win_w = args.winw
win_h = args.winh

# Tenta carregar nomes das classes a partir do data.yaml (se existir)
def load_names_from_yaml(dataset_train_path):
    # data.yaml esperado em GYU_DET/data.yaml (um nível acima de train)
    yaml_path = os.path.join(os.path.dirname(dataset_train_path), "data.yaml")
    if not os.path.exists(yaml_path):
        return None
    txt = open(yaml_path, "r", encoding="utf-8").read()
    m = re.search(r"^names:\s*(.+)$", txt, flags=re.M)
    if not m:
        return None
    s = m.group(1).strip()
    try:
        names = ast.literal_eval(s)
        return [str(x) for x in names]
    except Exception:
        # fallback simples
        s = s.strip("[]")
        items = [x.strip().strip('\"\'') for x in s.split(",") if x.strip()]
        return items

names = load_names_from_yaml(dataset_path)
if names is None:
    # fallback simples se não achar data.yaml
    names = [str(i) for i in range(100)]

# Gera cores por classe usando um colormap
num_classes = len(names)
colormap = cm.get_cmap("tab10")
class_colors = {i: tuple(int(255 * c) for c in colormap(i % 10)[:3]) for i in range(num_classes)}

# Pega 5 imagens de exemplo
# Pega lista completa e embaralha para amostrar aleatoriamente
all_images = sorted(os.listdir(img_dir))
random.shuffle(all_images)

# Seleciona imagens anotadas (ou inclui não anotadas se necessário)
selected = []
annotated = []
for img_name in all_images:
    if len(selected) >= num_images:
        break
    img_path = os.path.join(img_dir, img_name)
    lbl_path = os.path.join(lbl_dir, img_name.replace(".jpg", ".txt"))
    if only_annotated:
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            selected.append((img_name, img_path, lbl_path))
    else:
        selected.append((img_name, img_path, lbl_path if os.path.exists(lbl_path) else None))

# Se não encontrou suficientes anotadas e estamos pedindo apenas anotadas,
# completa com não-anotadas
if only_annotated and len(selected) < num_images:
    for img_name in all_images:
        if len(selected) >= num_images:
            break
        if any(img_name == s[0] for s in selected):
            continue
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, img_name.replace(".jpg", ".txt"))
        selected.append((img_name, img_path, lbl_path if os.path.exists(lbl_path) else None))

# Pega 5 imagens de exemplo
images = os.listdir(img_dir)[:5]

# Lista para guardar as imagens anotadas (RGB)
annotated = []

for img_name, img_path, lbl_path in selected:
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w, _ = img.shape

    if not lbl_path or not os.path.exists(lbl_path) or os.path.getsize(lbl_path) == 0:
        # se não há anotações, ainda assim adiciona imagem original
        annotated.append((img_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        continue

    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, x, y, bw, bh = map(float, parts)

            # YOLO -> pixels
            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)

            color = class_colors.get(int(cls) if int(cls) in class_colors else 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_text = names[int(cls)] if int(cls) < len(names) else str(int(cls))
            # fundo do texto para legibilidade
            ((tw, th), _) = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, label_text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    # Guardar imagem anotada (convertendo para RGB)
    annotated.append((img_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

# Mostrar todas as imagens em uma grade usando matplotlib, maximizando o uso da janela
if len(annotated) > 0:
    n = len(annotated)

    # calcula cols/rows para maximizar tamanho das imagens dentro da janela (considerando aspecto)
    aspect = float(win_w) / float(win_h) if win_h > 0 else 16/9
    cols = max(1, int((n * aspect) ** 0.5))
    cols = min(cols, n)
    rows = (n + cols - 1) // cols

    # calcula figsize em polegadas a partir de pixels e DPI
    dpi = plt.rcParams.get('figure.dpi', 100)
    fig_w = max(1.0, win_w / float(dpi))
    fig_h = max(1.0, win_h / float(dpi))

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=dpi)

    # tenta forçar o tamanho da janela do gerenciador (plataforma/backend dependent)
    try:
        manager = plt.get_current_fig_manager()
        # TkAgg
        try:
            manager.window.wm_geometry(f"{win_w}x{win_h}")
        except Exception:
            # Qt
            try:
                manager.window.resize(win_w, win_h)
            except Exception:
                pass
    except Exception:
        pass

    # normalizar axes para indexação uniforme
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    # remover margens para que as imagens ocupem o máximo possível
    # dar um pequeno espaço no topo/bottom para evitar corte de elementos do eixo
    plt.subplots_adjust(left=0.0, right=1.0, top=0.995, bottom=0.0, wspace=0.01, hspace=0.03)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            if idx < n:
                name, img_rgb = annotated[idx]
                ax.imshow(img_rgb, aspect='auto')
                # desenha o nome do arquivo sobre a imagem (top-left) com fundo semi-transparente
                ax.text(
                    0.01, 0.99, name,
                    transform=ax.transAxes,
                    fontsize=10,
                    color='white',
                    verticalalignment='top',
                    bbox=dict(facecolor='black', alpha=0.6, pad=2)
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')
            else:
                ax.axis('off')
            idx += 1

    plt.show()
