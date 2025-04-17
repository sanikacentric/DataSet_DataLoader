In any machine learning or deep learning project, managing your data pipeline is just as critical as designing the model itself. A strong foundation in handling datasets makes model training efficient and scalable.

In this article, we explore how to:

Create a synthetic dataset using Scikit-learn,

Prepare it using PyTorch utilities, and

Load it efficiently with a DataLoader for model training.

✨ Step 1: Creating a Synthetic Dataset
We use make_classification from sklearn.datasets to create a simple classification dataset:

python
Copy
Edit
from sklearn.datasets import make_classification
import torch

# Generate a synthetic dataset
X, y = make_classification(
    n_samples=10,       # 10 samples
    n_features=2,       # 2 features per sample
    n_informative=2,    # Both features are informative
    n_redundant=0,      # No redundant features
    n_classes=2,        # Binary classification
    random_state=42     # Reproducibility
)
This produces:

X: 10 rows × 2 columns (features),

y: Labels corresponding to each row (either 0 or 1).

✨ Step 2: Converting Data to Tensors
Since PyTorch models work with tensors, we must convert our X and y into PyTorch tensors:

python
Copy
Edit
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
This ensures our dataset is in the correct format for PyTorch processing.

✨ Step 3: Creating a Dataset Object
To leverage PyTorch's built-in DataLoader, we must wrap our tensors inside a custom Dataset:

python
Copy
Edit
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

dataset = CustomDataset(X_tensor, y_tensor)
This custom Dataset class:

Returns a sample (features, label) pair on every access,

Reports the length (number of samples).

✨ Step 4: Loading Data Efficiently with DataLoader
Finally, we load our dataset with the help of PyTorch’s DataLoader:

python
Copy
Edit
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

# Iterate through one epoch
for features, labels in dataloader:
    print(f"Features: {features}")
    print(f"Labels: {labels}")
Key features of DataLoader:

Batching: Groups samples into mini-batches.

Shuffling: Randomizes data order every epoch.

Parallel Loading: (using num_workers for larger datasets).

🎯 Conclusion
With just a few lines of code, you've:

Simulated a realistic classification dataset,

Converted it for PyTorch compatibility,

Built a custom Dataset,

Efficiently loaded it using DataLoader.

This pipeline is the backbone of real-world AI/ML projects, where data handling needs to be efficient, reproducible, and scalable.
