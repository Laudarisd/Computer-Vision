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

# Normalize the `ValveType1` column (strip spaces and convert to lowercase)
df["ValveType1"] = df["ValveType1"].str.strip().str.lower()

# Helper function to match and copy files to the appropriate folder
def copy_file(tag, dest_folder):
    prefixes = ["FV-", "PV-", "LV-", "XV-", "CV-", "AV-", "MV-", "HV-", "TV-", "PDV-", "SP-", "MOV-"]
    tag = str(tag).replace(" ", "").upper()
    match = None
    for prefix in prefixes:
        if prefix in tag:
            match = tag.split(prefix)[-1]
            break
    
    if not match:
        print(f"No matching prefix found for tag: {tag}")
        return
    
    pdf_name = [f for f in os.listdir(pdf_folder) if match in f]
    if pdf_name:
        os.makedirs(dest_folder, exist_ok=True)
        shutil.copy(os.path.join(pdf_folder, pdf_name[0]), os.path.join(dest_folder, pdf_name[0]))
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
    "mov": "mov_valve"  # Added mapping for "MOV"
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
