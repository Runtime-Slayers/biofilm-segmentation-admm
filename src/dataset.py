"""
Dataset utility functions.
Handles structuring, formatting, and preprocessing steps for the biofilm dataset.
"""

import os
import shutil


def create_structured_directory(source_dir: str, dest_dir: str, class_names: list[str]) -> int:
    """
    Reads a flat directory containing image files, infers their class tags from
    their filenames, and copies them to a structured target directory grouping
    them into class folders.

    Parameters:
    -----------
    source_dir : str
        Path to the flat directory containing raw images.
    dest_dir : str
        Path to the destination folder where class directories will be created.
    class_names : list of str
        The expected class tags to locate within image filenames.

    Returns:
    --------
    copied_count : int
        The total number of images successfully structured and copied.
    """
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)

    for class_name in class_names:
        os.makedirs(os.path.join(dest_dir, class_name), exist_ok=True)

    copied_count = 0
    for filename in os.listdir(source_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            # Infer class by checking if a class name is in the filename
            found_class = None
            for class_name in class_names:
                if class_name in filename:
                    found_class = class_name
                    break

            if found_class:
                source_path = os.path.join(source_dir, filename)
                dest_path = os.path.join(dest_dir, found_class, filename)
                shutil.copy(source_path, dest_path)
                copied_count += 1

    return copied_count
