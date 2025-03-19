import os
import shutil
import pandas as pd


def remove_non_image_files(folder_path: str) -> None:
    files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    for file in files:
        if not file.lower().endswith('.jpg'):  # Check if it's not a .jpg file
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)  # Delete the file
            print(f"Deleted: {file}")


def rename_files(folder_path: str, csv_path: str) -> None:
    # rename the .jpg files for the input images
    for filename in os.listdir(folder_path):
        if "_downsampled" in filename:
            new_filename = filename.replace("_downsampled", "")
            old_filename_path = os.path.join(folder_path, filename)
            new_filename_path = os.path.join(folder_path, new_filename)
            os.rename(old_filename_path, new_filename_path)
            print(f"Renamed: {filename} -> {new_filename}")

    # rename the entries in the according .csv file
    df = pd.read_csv(csv_path)
    df['image'] = df['image'].str.replace('_downsampled', '', regex=False)

    # Save the modified DataFrame back
    df.to_csv(csv_path, index=False)


def train_val_test_split(folder_path: str, distribution: tuple[float, float, float], csv_path) -> None:
    # check if split sums up to 1.0
    assert sum(distribution) == 1.0, f"sum fo distribution should be 1.0 but is {sum(distribution)}"

    # split according to percentages -> maybe 80%, 10%, 10%
    count = len(os.listdir(folder_path))
    distribution_count = [int(d * count) for d in distribution]
    train_split = distribution_count[0]
    val_split = distribution_count[0] + distribution_count[1]

    # Split input images
    # Create train, val, test subfolders
    subfolders = ["train", "val", "test"]
    subfolder_paths = [os.path.join(folder_path, sf) for sf in subfolders]
    train_folder_path = subfolder_paths[0]
    val_folder_path = subfolder_paths[1]
    test_folder_path = subfolder_paths[2]

    for subfolder in subfolder_paths:
        os.makedirs(subfolder, exist_ok=True)

    # move images based on index to according folders
    files = sorted([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])
    for idx, file in enumerate(files):
        src_path = os.path.join(folder_path, file)

        if idx < train_split:
            dest_path = os.path.join(train_folder_path, file)
        elif idx < val_split:
            dest_path = os.path.join(val_folder_path, file)
        else:
            dest_path = os.path.join(test_folder_path, file)

        shutil.move(src_path, dest_path)
        print(f"Moved: {file} -> {dest_path}")

    # Split csv file
    df = pd.read_csv(csv_path)

    # Define the split ranges
    splits = [
        (0, train_split-1, "gt_train.csv"),
        (train_split, val_split-1, "gt_val.csv"),
        (val_split, len(df), "gt_test.csv"),
    ]

    # Create each split and save as a new CSV file
    idx = 0
    for start, end, filename in splits:
        chunk = df.iloc[start:end + 1]  # Select rows from start to end
        output_path = f"{subfolder_paths[idx]}/{filename}"  # Change to your desired output path
        chunk.to_csv(output_path, index=False)  # Save with header
        idx += 1
        print(f"Saved {output_path} ({start}-{end})")


def main() -> None:
    folder_path = "data/ISIC_2019_Training_Input"
    csv_path = "data/ISIC_2019_Training_GroundTruth.csv"
    remove_non_image_files(folder_path)
    rename_files(folder_path, csv_path)
    dist = (0.8, 0.1, 0.1)
    train_val_test_split(folder_path, dist, csv_path)
    processed = "data/ISIC_2019"
    os.rename(folder_path, processed)


if __name__ == "__main__":
    main()