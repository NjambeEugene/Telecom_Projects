from pathlib import Path

def fix_labels_v2(label_dir):
    """
    Remap class IDs:
    - Class 4 → Class 0 (car)
    - Class 15 → Class 1 (car_white)
    """
    label_path = Path(label_dir)
    fixed_count = 0
    
    class_mapping = {
        4: 0,    # car
        15: 1    # car_white
    }
    
    for txt_file in label_path.rglob("*.txt"):
        try:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            
            modified = False
            new_lines = []
            
            for line in lines:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    
                    # Map the class ID
                    if class_id in class_mapping:
                        parts[0] = str(class_mapping[class_id])
                        modified = True
                    
                    new_lines.append(' '.join(parts) + '\n')
            
            if modified:
                with open(txt_file, 'w') as f:
                    f.writelines(new_lines)
                fixed_count += 1
        
        except Exception as e:
            print(f"Error processing {txt_file}: {e}")
    
    return fixed_count

if __name__ == "__main__":
    print("Remapping labels...")
    print("Class 4 → 0 (car)")
    print("Class 15 → 1 (car_white)")
    
    train_labels = r"YOLO/labels/train"
    print(f"\nFixing training labels...")
    count1 = fix_labels_v2(train_labels)
    print(f"Fixed {count1} files")
    
    val_labels = r"YOLO/labels/val"
    print(f"\nFixing validation labels...")
    count2 = fix_labels_v2(val_labels)
    print(f"Fixed {count2} files")
    
    print("\nRemapping complete! Now run analyze_labels.py to verify.")
