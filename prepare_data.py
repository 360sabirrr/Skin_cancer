import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

print("🚀 Started Preparing Dataset...")

# ================= ABSOLUTE PATHS =================
base_dir = r"C:\Users\sabir\OneDrive\Desktop\Skin_cancer_detection_system"
img_part1 = os.path.join(base_dir, "Dataset", "HAM10000_images_part_1")
img_part2 = os.path.join(base_dir, "Dataset", "HAM10000_images_part_2")
merged_path = os.path.join(base_dir, "Dataset", "images")
metadata_csv = os.path.join(base_dir, "Dataset", "HAM10000_metadata.csv")

train_path = os.path.join(base_dir, "Data", "train")
val_path = os.path.join(base_dir, "Data", "val")

# ================= STEP 0: VERIFY SOURCE =================
print("Checking source folders...")
for p in [img_part1, img_part2]:
    if not os.path.exists(p):
        raise Exception(f"❌ Folder does not exist: {p}")
print("✅ Source folders exist")

# ================= STEP 1: MERGE IMAGES =================
os.makedirs(merged_path, exist_ok=True)
valid_ext = ('.jpg', '.jpeg', '.png')

count = 0
for folder in [img_part1, img_part2]:
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(valid_ext):
                src = os.path.join(root, file)
                dst = os.path.join(merged_path, file)
                if not os.path.exists(dst):
                    shutil.copy(src, dst)
                    count += 1
print(f"✅ Images merged: {count}")

# ================= STEP 2: LOAD METADATA =================
df = pd.read_csv(metadata_csv)
print("📊 Class distribution:\n", df['dx'].value_counts())

# ================= STEP 3: TRAIN-VAL SPLIT =================
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df['dx'], random_state=42
)

# ================= STEP 4: CREATE CLASS FOLDERS =================
def create_dataset(dataframe, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    for label in dataframe['dx'].unique():
        os.makedirs(os.path.join(target_folder, label), exist_ok=True)
    
    copied = 0
    for _, row in dataframe.iterrows():
        img_name = row['image_id'] + ".jpg"
        src = os.path.join(merged_path, img_name)
        dst = os.path.join(target_folder, row['dx'], img_name)
        if os.path.exists(src):
            shutil.copy(src, dst)
            copied += 1
    print(f"✅ {target_folder} -> {copied} images copied")

create_dataset(train_df, train_path)
create_dataset(val_df, val_path)

# ================= STEP 5: FINAL VERIFICATION =================
def count_images(folder):
    return sum(len(files) for _, _, files in os.walk(folder))

train_count = count_images(train_path)
val_count = count_images(val_path)
total = train_count + val_count

print("\n🎯 FINAL COUNT:")
print("Train:", train_count)
print("Validation:", val_count)
print("Total:", total)
print("\n✅ DATASET READY SUCCESSFULLY!")