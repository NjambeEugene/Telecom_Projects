import os
from pathlib import Path

print("Searching for trained model file...")

# Search for best.pt
for root, dirs, files in os.walk(r"runs"):
    for file in files:
        if file == "best.pt":
            full_path = os.path.join(root, file)
            print(f"Found: {full_path}")

# Also check all .pt files
print("\nAll .pt files found:")
for root, dirs, files in os.walk(r"runs"):
    for file in files:
        if file.endswith(".pt"):
            full_path = os.path.join(root, file)
            print(f"  {full_path}")

# List runs directory structure
print("\nRuns directory structure:")
for item in Path("runs").rglob("*"):
    if item.is_file():
        print(f"  {item}")
