import os
import json
import random
from pathlib import Path
from shutil import copyfile, move
import yaml

def convert_coco_to_yolo(coco_json_path, output_dir, val_count=5):
    # Create directories for YOLO format
    images_dir = Path(output_dir) / 'images'
    labels_dir = Path(output_dir) / 'labels'
    images_train_dir = images_dir / 'train'
    labels_train_dir = labels_dir / 'train'
    images_val_dir = images_dir / 'val'
    labels_val_dir = labels_dir / 'val'
    images_train_dir.mkdir(parents=True, exist_ok=True)
    labels_train_dir.mkdir(parents=True, exist_ok=True)
    images_val_dir.mkdir(parents=True, exist_ok=True)
    labels_val_dir.mkdir(parents=True, exist_ok=True)
    
    # Load COCO data
    with open(coco_json_path) as f:
        coco_data = json.load(f)
    
    images_info = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    category_names = [categories[cat_id] for cat_id in sorted(categories.keys())]

    all_image_ids = list(images_info.keys())
    random.shuffle(all_image_ids)
    
    # Select initial validation images
    val_image_ids = set(all_image_ids[:val_count])
    train_image_ids = set(all_image_ids[val_count:])

    # Dictionary to track image and label file paths for validation
    val_image_files = set()
    
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        img_info = images_info.get(image_id)
        
        if img_info is None:
            print(f"Warning: Image ID {image_id} not found in images_info.")
            continue
        
        img_width = img_info['width']
        img_height = img_info['height']
        
        category_id = ann['category_id'] - 1  # YOLO class ids should start from 0
        bbox = ann['bbox']
        
        # COCO bbox format: [top left x, top left y, width, height]
        x, y, w, h = bbox
        x_center = x + w / 2
        y_center = y + h / 2
        
        # Normalize values
        x_center /= img_width
        y_center /= img_height
        w /= img_width
        h /= img_height
        
        # YOLO format: <class_id> <x_center> <y_center> <width> <height>
        yolo_line = f"{category_id} {x_center} {y_center} {w} {h}\n"
        
        # Save to label file
        img_filename = img_info['file_name']
        label_filename = Path(img_filename).stem + '.txt'
        if image_id in val_image_ids:
            label_file_path = labels_val_dir / label_filename
            val_image_files.add(img_filename)  # Track validation image files
        else:
            label_file_path = labels_train_dir / label_filename
        
        with open(label_file_path, 'a') as label_file:
            label_file.write(yolo_line)
        
        # Copy image to images directory
        src_img_path = Path(coco_json_path).parent.parent / 'images' / img_filename
        if not src_img_path.exists():
            print(f"Warning: Source image file {src_img_path} does not exist.")
            continue
        
        if image_id in val_image_ids:
            dest_img_path = images_val_dir / img_filename
        else:
            dest_img_path = images_train_dir / img_filename
        
        if not dest_img_path.exists():
            try:
                copyfile(src_img_path, dest_img_path)
            except IOError as e:
                print(f"Error copying file {src_img_path} to {dest_img_path}: {e}")

    # At the end, move additional random images from train to val
    remaining_image_files = list(train_image_ids)
    random.shuffle(remaining_image_files)
    additional_val_image_files = set(remaining_image_files[:val_count])

    for img_id in additional_val_image_files:
        img_info = images_info.get(img_id)
        
        if img_info is None:
            print(f"Warning: Image ID {img_id} not found in images_info.")
            continue
        
        img_filename = img_info['file_name']
        src_img_path = images_train_dir / img_filename
        dest_img_path = images_val_dir / img_filename

        if not src_img_path.exists():
            print(f"Warning: Source image file {src_img_path} does not exist.")
            continue

        if not dest_img_path.exists():
            try:
                move(src_img_path, dest_img_path)  # Move the image file
            except IOError as e:
                print(f"Error moving file {src_img_path} to {dest_img_path}: {e}")

        # Move the corresponding label file
        label_filename = Path(img_filename).stem + '.txt'
        src_label_path = labels_train_dir / label_filename
        dest_label_path = labels_val_dir / label_filename

        if src_label_path.exists():
            try:
                move(src_label_path, dest_label_path)  # Move the label file
            except IOError as e:
                print(f"Error moving label file {src_label_path} to {dest_label_path}: {e}")

    return category_names

def create_data_yaml(output_path, train_dir, val_dir, class_names):
    data = {
        'train': str(train_dir),
        'val': str(val_dir),
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

# Main execution
coco_json_path = '/content/dataset/annotations/instances_Train.json'
output_dir = '/content/dataset/yolov8_dataset'
train_dir = '/content/dataset/yolov8_dataset/images/train'
val_dir = '/content/dataset/yolov8_dataset/images/val'

# Convert dataset and get class names
class_names = convert_coco_to_yolo(coco_json_path, output_dir, val_count=5)

# Create data.yaml file
output_yaml_path = Path(output_dir) / 'data.yaml'
create_data_yaml(output_yaml_path, train_dir, val_dir, class_names)
