import os
import shutil
import pandas as pd

# Load the Excel data
excel_file = "all_data.xlsx"  # Replace with your actual Excel file path
pdf_folder = "combined"  # Folder where PDF files are stored
output_folder = "categorized_files"  # Folder to save categorized files
os.makedirs(output_folder, exist_ok=True)

# Read the Excel file
df = pd.read_excel(excel_file)

# Normalize columns
df["ValveType1"] = df["ValveType1"].str.strip().str.lower()
df[" ValveType2"] = df[" ValveType2"].str.strip()

# List all files in the folder for debugging
pdf_files = os.listdir(pdf_folder)
print("Files in PDF folder:", pdf_files)

# Helper function to match and copy files to the appropriate folder
def copy_file(tag, dest_folder):
    tag = str(tag).strip().upper()  # Normalize the tag
    print(f"Processing tag: {tag}")  # Debugging

    # Match the entire tag in the filename
    matched_files = [f for f in pdf_files if tag in f.upper()]
    if matched_files:
        print(f"Found matching files for tag {tag}: {matched_files}")
        os.makedirs(dest_folder, exist_ok=True)
        shutil.copy(os.path.join(pdf_folder, matched_files[0]), os.path.join(dest_folder, matched_files[0]))
    else:
        print(f"No matching PDF found for tag: {tag}")

# 1. Categorize by Actuator Type
cv_cyl_folder = os.path.join(output_folder, "cv_cyl")
for _, row in df.iterrows():
    subfolder = os.path.join(cv_cyl_folder, str(row["ActuatorType"]).strip())
    copy_file(row["Tag"], subfolder)

# 2. Categorize by Valve Type 1
valve_type_1_folder = os.path.join(output_folder, "valve_type_1")
valve_type_1_mapping = {
    "control valve": "control_valve",
    "on-off valve": "on_off_valve",
    "mov": "mov_valve"  # Added missing mapping
}
for _, row in df.iterrows():
    valve_type = row["ValveType1"]
    if valve_type in valve_type_1_mapping:
        subfolder = os.path.join(valve_type_1_folder, valve_type_1_mapping[valve_type])
        copy_file(row["Tag"], subfolder)
    else:
        print(f"Unmapped Valve Type 1: {valve_type}")

# 3. Categorize by Valve Type 2
valve_type_2_folder = os.path.join(output_folder, "valve_type_2")
for _, row in df.iterrows():
    valve_type_2 = str(row[" ValveType2"]).strip()
    subfolder = os.path.join(valve_type_2_folder, valve_type_2)
    copy_file(row["Tag"], subfolder)

print("Files categorized successfully!")
