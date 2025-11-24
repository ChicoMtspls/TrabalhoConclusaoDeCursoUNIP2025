import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
import random

######split_dataset######
# Esse script divide o dataset CODEBRIM (que não veio separado),
# em pastas de treino, validacao e teste (train,val,test)
#########################


# seed para reproduzir resultados aleatorios
SEED = 42
random.seed(SEED)

# Localização dos arquivos #MUDAR PELO MENOS OS DIRETORIOS ANTES DE RODAR
SOURCE_DIR = r"D:\TCC\datasets\datasetsNovos\CODEBRIM_split"
OUTPUT_DIR = r"D:\TCC\datasets\datasetsNovos\CODEBRIM_split_yolo"
IMAGES_SOURCE = os.path.join(SOURCE_DIR, "images")
LABELS_SOURCE = os.path.join(SOURCE_DIR, "labels")
CLASSES_FILE = os.path.join(LABELS_SOURCE, "classes.txt")

# separa na proporção desejada (70-15-15 tá bom)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def read_classes():
    #lê o nome dentro do classes.txt, que vem especificamente no CODEBRIM
    with open(CLASSES_FILE, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return classes

def create_directory_structure():
    #cria os diretorios
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)
    print(f"Diretórios criados em: {OUTPUT_DIR}")

def get_paired_files():
    #Pega a lista de pares de arquivos image e label
    image_files = [f for f in os.listdir(IMAGES_SOURCE) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    
    # filtra só os que tem label correspondente
    paired_files = []
    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        if os.path.exists(os.path.join(LABELS_SOURCE, label_file)):
            paired_files.append(img_file)
    
    return paired_files

def split_dataset(files):
    #separa os arquivos em si (de acordo com a proporção definida)
    # separa os de treino primeiro, que costumam ser a maior parte
    train_files, temp_files = train_test_split(
        files,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=SEED
    )
    
    # separa o restante entre validação e teste
    val_files, test_files = train_test_split(
        temp_files,
        test_size=0.5,
        random_state=SEED
    )
    
    return train_files, val_files, test_files

def copy_files_to_splits(train_files, val_files, test_files):
    #copia as imagens e labels para as pastas respectivas
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, file_list in splits.items():
        for img_file in file_list:
            # Copy image
            src_img = os.path.join(IMAGES_SOURCE, img_file)
            dst_img = os.path.join(OUTPUT_DIR, split_name, 'images', img_file)
            shutil.copy2(src_img, dst_img)
            
            # Copy label
            label_file = os.path.splitext(img_file)[0] + '.txt'
            src_label = os.path.join(LABELS_SOURCE, label_file)
            dst_label = os.path.join(OUTPUT_DIR, split_name, 'labels', label_file)
            shutil.copy2(src_label, dst_label)
        
        print(f"Copiados {len(file_list)} arquivos para o split: {split_name}")

def create_data_yaml(classes):
    #cria a data.yaml pro yolo conseguir ler o dataset
    dataset_info = {
        'path': OUTPUT_DIR,
        'train': os.path.join(OUTPUT_DIR, 'train', 'images'),
        'val': os.path.join(OUTPUT_DIR, 'val', 'images'),
        'test': os.path.join(OUTPUT_DIR, 'test', 'images'),
        'nc': len(classes),
        'names': {i: class_name for i, class_name in enumerate(classes)}
    }
    
    yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_info, f, default_flow_style=False, sort_keys=False)
    
    print(f"data.yaml criado em {yaml_path}")
    return yaml_path

def main():
    print("=" * 60)
    print("SEPARAÇÃO DO DATASET CODEBRIM PARA YOLO")
    print("=" * 60)
    
    # lê as classes
    classes = read_classes()
    print(f"\n{len(classes)} classes encontradas:")
    for i, class_name in enumerate(classes):
        print(f"   {i}: {class_name}")
    
    # Cria os diretórios
    print(f"\nCriando os diretórios de saida...")
    create_directory_structure()
    
    # Pega os arquivos em par (imagem e label)
    print(f"\nEncontrando pares image-label...")
    paired_files = get_paired_files()
    print(f"{len(paired_files)} pares image-label encontrados.")
    
    # separa o dataset
    print(f"\nSeparando dataset: ({TRAIN_RATIO*100:.0f}% train, {VAL_RATIO*100:.0f}% val, {TEST_RATIO*100:.0f}% test)...")
    train_files, val_files, test_files = split_dataset(paired_files)
    print(f"   Train: {len(train_files)} files")
    print(f"   Val:   {len(val_files)} files")
    print(f"   Test:  {len(test_files)} files")
    
    # copia os arquivos para as pastas respectivas
    print(f"\nCopiando arquivos para as pastas respectivas...")
    copy_files_to_splits(train_files, val_files, test_files)
    
    # cria o data.yaml
    print(f"\nCriando o data.yaml para o YOLO...")
    create_data_yaml(classes)
    
    print("\n" + "=" * 60)
    print("CONCLUIDO!")
    print("=" * 60)
    print(f"\nDiretório final: {OUTPUT_DIR}")
    print("\nComando provável:")
    print(f'   yolo detect train data={os.path.join(OUTPUT_DIR, "data.yaml")} model=yolov8n.pt epochs=100')
    print("=" * 60)

if __name__ == "__main__":
    main()