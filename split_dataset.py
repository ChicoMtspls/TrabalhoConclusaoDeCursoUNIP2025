import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
import random

# Set seed for reproducibility
SEED = 42
random.seed(SEED)

# Configuration
SOURCE_DIR = r"D:\TCC\datasets\datasetsNovos\CODEBRIM_split"
OUTPUT_DIR = r"D:\TCC\datasets\datasetsNovos\CODEBRIM_split_yolo"
IMAGES_SOURCE = os.path.join(SOURCE_DIR, "images")
LABELS_SOURCE = os.path.join(SOURCE_DIR, "labels")
CLASSES_FILE = os.path.join(LABELS_SOURCE, "classes.txt")

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def read_classes():
    """Read class names from classes.txt"""
    with open(CLASSES_FILE, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return classes

def create_directory_structure():
    """Create train/val/test directory structure"""
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)
    print(f"✓ Created directory structure at {OUTPUT_DIR}")

def get_paired_files():
    """Get list of image-label pairs that exist"""
    image_files = [f for f in os.listdir(IMAGES_SOURCE) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    
    # Filter to only images that have corresponding labels
    paired_files = []
    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        if os.path.exists(os.path.join(LABELS_SOURCE, label_file)):
            paired_files.append(img_file)
    
    return paired_files

def split_dataset(files):
    """Split files into train/val/test"""
    # First split: 70% train, 30% temp (which will be split into val/test)
    train_files, temp_files = train_test_split(
        files,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=SEED
    )
    
    # Second split: split the 30% into val and test (50/50 = 15% each)
    val_files, test_files = train_test_split(
        temp_files,
        test_size=0.5,
        random_state=SEED
    )
    
    return train_files, val_files, test_files

def copy_files_to_splits(train_files, val_files, test_files):
    """Copy image and label files to respective split folders"""
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
        
        print(f"✓ Copied {len(file_list)} files to {split_name} split")

def create_data_yaml(classes):
    """Create data.yaml file for YOLO training"""
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
    
    print(f"✓ Created data.yaml at {yaml_path}")
    return yaml_path

def main():
    print("=" * 60)
    print("CODEBRIM Dataset Splitter for YOLO")
    print("=" * 60)
    
    # Read classes
    classes = read_classes()
    print(f"\n📋 Found {len(classes)} classes:")
    for i, class_name in enumerate(classes):
        print(f"   {i}: {class_name}")
    
    # Create directory structure
    print(f"\n📁 Creating output directory structure...")
    create_directory_structure()
    
    # Get paired files
    print(f"\n🔍 Finding image-label pairs...")
    paired_files = get_paired_files()
    print(f"✓ Found {len(paired_files)} paired image-label files")
    
    # Split dataset
    print(f"\n✂️ Splitting dataset ({TRAIN_RATIO*100:.0f}% train, {VAL_RATIO*100:.0f}% val, {TEST_RATIO*100:.0f}% test)...")
    train_files, val_files, test_files = split_dataset(paired_files)
    print(f"   Train: {len(train_files)} files")
    print(f"   Val:   {len(val_files)} files")
    print(f"   Test:  {len(test_files)} files")
    
    # Copy files
    print(f"\n📋 Copying files to split folders...")
    copy_files_to_splits(train_files, val_files, test_files)
    
    # Create data.yaml
    print(f"\n⚙️ Creating data.yaml for YOLO...")
    create_data_yaml(classes)
    
    print("\n" + "=" * 60)
    print("✅ Dataset split completed successfully!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nYour YOLO training command should look like:")
    print(f'   yolo detect train data={os.path.join(OUTPUT_DIR, "data.yaml")} model=yolov8n.pt epochs=100')
    print("=" * 60)

if __name__ == "__main__":
    main()
