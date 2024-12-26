import os
import shutil

def compare_files(first_dir, second_dir, copy_extra_file, first_exts, second_exts):
    """
    Compare files by name and count in two directories and handle extra files.

    Args:
        first_dir (str): Path to the first directory.
        second_dir (str): Path to the second directory.
        copy_extra_file (str): Path to save extra files found in either directory.
        first_exts (tuple): Extensions to consider in the first directory.
        second_exts (tuple): Extensions to consider in the second directory.
    """
    # Parse files in the first directory and remove extensions
    first_files = [os.path.splitext(file)[0] for file in os.listdir(first_dir) if file.endswith(first_exts)]

    # Parse files in the second directory and remove extensions
    second_files = [os.path.splitext(file)[0] for file in os.listdir(second_dir) if file.endswith(second_exts)]

    # Compare file counts
    if len(first_files) > len(second_files):
        print(f"More files in {first_dir} than {second_dir}: {len(first_files)} > {len(second_files)}")
        print("Extra files in the first directory:")
        for file in first_files:
            if file not in second_files:
                print(file)
                for ext in first_exts:
                    file_path = f"{first_dir}/{file}{ext}"
                    if os.path.exists(file_path):
                        shutil.copy(file_path, f"{copy_extra_file}/{file}{ext}")
                        break
    elif len(first_files) < len(second_files):
        print(f"More files in {second_dir} than {first_dir}: {len(second_files)} > {len(first_files)}")
        print("Extra files in the second directory:")
        for file in second_files:
            if file not in first_files:
                print(file)
                for ext in second_exts:
                    file_path = f"{second_dir}/{file}{ext}"
                    if os.path.exists(file_path):
                        shutil.copy(file_path, f"{copy_extra_file}/{file}{ext}")
                        break
    else:
        print(f"Equal number of files in both directories: {len(first_files)}")

if __name__ == '__main__':
    first_dir = './all_images'
    second_dir = './main_xmls'
    copy_extra_file = './required_copy/'
    first_exts = ('.png', '.jpg', '.jpeg')  # Specify allowed extensions for the first directory
    second_exts = ('.xml', '.txt')  # Specify allowed extensions for the second directory

    os.makedirs(copy_extra_file, exist_ok=True)
    compare_files(first_dir, second_dir, copy_extra_file, first_exts, second_exts)
