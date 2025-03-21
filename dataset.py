import os
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image


def init_filenames(folder_path: str) -> list[str]:
    filenames =  [os.path.splitext(filename)[0] for filename in os.listdir(folder_path) if filename.endswith(".jpg")]
    filenames = sorted(filenames)
    return filenames


def init_dataframe(folder_path: str) -> pd.DataFrame:
    dataframe_path = f"{folder_path}/gt_train.csv"
    return pd.read_csv(dataframe_path)


class MelanomaImgClassPair(Dataset):
    """
        Dataset for the ISIC 2019 melanoma classification task.
        Given images of skin and a csv file labeling each image to a melanoma class,
        returns image as 3d pytorch tensor and classification as one hot vector representation.

        Args:
            folder_path (str): The path to the folder containing the images and the csv file
    """
    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path
        self.filenames = init_filenames(folder_path)
        self.dataframe = init_dataframe(folder_path)
        self.classnames = self.dataframe.columns[1:].tolist()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        # load image as tensor
        img = cv2.imread(f"{self.folder_path}/{self.filenames[idx]}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transforms.ToTensor()(img)

        # load corresponding csv entry (through filename) as tensor
        row = self.dataframe[self.dataframe["image"] == self.filenames[idx]]
        assert not row.empty, f"Image {self.filenames[idx]} not found in csv file!"
        label_values = row.iloc[:, 1:].values.flatten().astype(int)
        gt_class = torch.tensor(label_values, dtype=torch.int64)

        return img, gt_class

    def visualize_item(self, idx: int):
        """
            Displays the image and the corresponding label.
            This is used to verify if our dataset class works properly.

            Args:
                idx (int): Index used to load the image and label.
        """
        img, label = self.__getitem__(idx)
        # Convert image tensor to PIL Image
        image = to_pil_image(img)

        # Get the class name from the one-hot encoded vector
        class_index = torch.argmax(label).item()
        class_name = self.classnames[class_index]

        # Plot the image
        plt.figure(figsize=(4, 4))
        plt.imshow(image)
        plt.title(f"File: {self.filenames[idx]}, Class: {class_name}")
        plt.axis("off")  # Hide axes
        plt.show()


def main() -> None:
    folder_path = "data/ISIC_2019/train"
    isic_dataset = MelanomaImgClassPair(folder_path)
    for idx in range(len(isic_dataset)):
        isic_dataset.visualize_item(idx)


if __name__ == "__main__":
    main()
