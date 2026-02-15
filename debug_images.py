"""
Debug script to find actual image files
"""
import os
import pandas as pd

# Check what's in the images directory
image_dir = "data/images_gz2/images"
print(f"Checking directory: {image_dir}")
print(f"Directory exists: {os.path.exists(image_dir)}")

if os.path.exists(image_dir):
    files = os.listdir(image_dir)[:20]
    print(f"\nFirst 20 files in directory:")
    for f in files:
        print(f"  {f}")
    
    # Check file extensions
    extensions = {}
    for f in os.listdir(image_dir):
        ext = os.path.splitext(f)[1].lower()
        extensions[ext] = extensions.get(ext, 0) + 1
    
    print(f"\nFile extensions found:")
    for ext, count in extensions.items():
        print(f"  {ext}: {count} files")

# Check CSV for object IDs
csv_path = "data/gz2_hart16.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, nrows=5)
    print(f"\nFirst 5 object IDs from CSV:")
    print(df['dr7objid'].values)
    
    # Check if these files exist
    print(f"\nChecking if image files exist:")
    for obj_id in df['dr7objid'].values[:5]:
        jpg_path = os.path.join(image_dir, f"{obj_id}.jpg")
        png_path = os.path.join(image_dir, f"{obj_id}.png")
        print(f"  {obj_id}.jpg exists: {os.path.exists(jpg_path)}")
        print(f"  {obj_id}.png exists: {os.path.exists(png_path)}")
