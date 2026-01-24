from pathlib import Path
from collections import Counter

def analyze_labels(label_dir):
    """Analyze label class distribution"""
    class_counts = Counter()
    
    for txt_file in Path(label_dir).rglob("*.txt"):
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
    
    return class_counts

# Analyze training labels
train_labels = r"YOLO/labels/train"
train_counts = analyze_labels(train_labels)

print("Training Label Distribution:")
print(f"  Class 0 (car): {train_counts.get(0, 0)}")
print(f"  Class 1 (car_white): {train_counts.get(1, 0)}")
print(f"  Other classes: {sum(v for k,v in train_counts.items() if k > 1)}")
print(f"  Total: {sum(train_counts.values())}")

# Analyze validation labels
val_labels = r"YOLO/labels/val"
val_counts = analyze_labels(val_labels)

print("\nValidation Label Distribution:")
print(f"  Class 0 (car): {val_counts.get(0, 0)}")
print(f"  Class 1 (car_white): {val_counts.get(1, 0)}")
print(f"  Other classes: {sum(v for k,v in val_counts.items() if k > 1)}")
print(f"  Total: {sum(val_counts.values())}")

# Show all unique classes
all_classes = set(train_counts.keys()) | set(val_counts.keys())
print(f"\nAll class IDs found: {sorted(all_classes)}")
