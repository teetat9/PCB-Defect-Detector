import os
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class PCBDataLoader:
    def __init__(self, data_path, img_size=(224, 224)):
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.defect_types = ["missing_hole", "open_circuit", "short", "spur", "spurious_copper", "mouse_bite"]
        self.label_map = {defect_type: idx for idx, defect_type in enumerate(self.defect_types)}

    def load_data(self):
        """Load all images and its labels"""
        images = []
        labels = []

        for defect_type in self.defect_types:
            defect_path = self.data_path / defect_type # defect directory
            for img_file in defect_path.glob("*.jpg"):
                # Load and resize all images
                img = cv2.imread(img_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)

                images.append(img)
                labels.append(self.label_map[defect_type])

        return np.array(images), np.array(labels)
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and validation sets"""
        images, labels = self.load_data()

        X_train, X_test, y_train, y_test = train_test_split(images, 
                                                            labels, 
                                                            test_size=test_size,
                                                            random_state=random_state,
                                                            stratify=labels # Ensure balance split between classes
                                                            )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_test)}")
        print(f"Class distribution (train): ")
        for defect, idx in self.label_map.items():
            count = np.sum(y_train == idx)
            print(f"    {defect}: {count}")
        
        return X_train, X_test, y_train, y_test

    def visualize_batch(self, images, labels, num_samples=9):
        """Visualize batch of images"""
        num_samples = min(num_samples, len(images))
        rows = int(np.sqrt(num_samples))
        columns = int(np.ceil(num_samples / rows)) # Or (num_samples + rows - 1) // rows to immitate ceil division

        fig, axes = plt.subplots(rows, columns, figsize=(12, 12))
        axes = axes.flatten() if num_samples > 1 else [axes]

        for idx in range(num_samples):
            axes[idx].imshow(images[idx])
            label_name = self.defect_types[labels[idx]]
            axes[idx].set_title(label_name)
            axes[idx].axis('off')
        
        # Hide extra subplots
        for idx in range(num_samples, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()


# Run the file
if __name__ == "__main__":
    # Initialize data loader
    loader = PCBDataLoader("organized_dataset")
    # Load and split data
    X_train, X_test, y_train, y_test = loader.split_data()

    # Visualize some training examples
    print("\nVisualizing sample batch...")
    indices = np.random.choice(len(X_train), 9, replace=False)

    # # One image per class (6 samples)
    # indices = []
    # for idx in range(len(loader.defect_types)):
    #     indices_of_this_class = [i for i, label in enumerate(y_train) if label == idx]
    #     random_index = np.random.choice(indices_of_this_class)
    #     indices.append(random_index)

    loader.visualize_batch(X_train[indices], y_train[indices], num_samples=6)
