import os
from pathlib import Path

def fix_labels(label_dir, class_mapping=None):
    """
    Fix label files by remapping incorrect class IDs to valid ones.
    
    Args:
        label_dir: Path to labels directory
        class_mapping: Dict mapping old class IDs to new ones
                      If None, maps all non-zero classes to 1, keeps 0 as is
    """
    if class_mapping is None:
        # Default: 0->0 (car), everything else->1 (car_white)
        class_mapping = {4: 1, 15: 1, 2: 1, 3: 1, 5: 1}  # Add more as needed
    
    label_path = Path(label_dir)
    fixed_count = 0
    
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
                    elif class_id > 1:
                        # Any unknown class > 1 -> class 1
                        parts[0] = str(1)
                        modified = True
                    
                    new_lines.append(' '.join(parts) + '\n')
            
            if modified:
                with open(txt_file, 'w') as f:
                    f.writelines(new_lines)
                fixed_count += 1
                print(f"Fixed: {txt_file}")
        
        except Exception as e:
            print(f"Error processing {txt_file}: {e}")
    
    print(f"\nTotal files fixed: {fixed_count}")

if __name__ == "__main__":
    # Fix training labels
    train_labels = r"C:\Users\NJAMBE\Documents\TRAFFIC\YOLO\labels\train"
    print("Fixing training labels...")
    fix_labels(train_labels)
    
    # Fix validation labels
    val_labels = r"C:\Users\NJAMBE\Documents\TRAFFIC\YOLO\labels\val"
    print("\nFixing validation labels...")
    fix_labels(val_labels)
    
    print("\nLabel fixing complete!")
