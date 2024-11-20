import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image

class TestPairsDataset(Dataset):
    def __init__(self, txt_file, image_dir, transform=None):
        """
        Args:
            txt_file (str): Path to the text file with image pairs.
            image_dir (str): Directory containing all images.
            transform (callable, optional): Transform to apply to images.
        """
        self.image_dir = image_dir
        self.transform = transform

        # Read pairs from text file
        with open(txt_file, 'r') as file:
            self.pairs = [line.strip().split() for line in file.readlines()]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr_name, hr_name = self.pairs[idx]
        lr_path = os.path.join(self.image_dir, lr_name + '.png')  # Assuming PNG format
        hr_path = os.path.join(self.image_dir, hr_name + '.png')

        # Load images
        lr_image = Image.open(lr_path).convert('RGB')
        hr_image = Image.open(hr_path).convert('RGB')

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize

# Define transform
transform = Compose([
    Resize((128, 128)),  # Resize images to match model input size
    ToTensor()           # Convert to PyTorch tensors
])

# Initialize dataset and dataloader
test_dataset = TestPairsDataset(
    txt_file='../../RRSSRDset/test_pairs.txt',
    image_dir='../..RRSSRDset/',
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def test_model_with_pairs(model, dataloader, device):
    model.eval()
    results = []

    with torch.no_grad():
        for lr_image, hr_image in dataloader:
            lr_image = lr_image.to(device)
            hr_image = hr_image.to(device)

            # Get model output
            output = model(lr_image, hr_image)

            # Collect results (e.g., save or calculate metrics)
            results.append((output.cpu(), hr_image.cpu()))

    return results

# Run testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DATSR().to(device)  # Assuming your model is initialized
results = test_model_with_pairs(model, test_loader, device)


def test_model_with_pairs(model, dataloader, device):
    model.eval()
    results = []

    with torch.no_grad():
        for lr_image, hr_image in dataloader:
            lr_image = lr_image.to(device)
            hr_image = hr_image.to(device)

            # Get model output
            output = model(lr_image, hr_image)

            # Collect results (e.g., save or calculate metrics)
            results.append((output.cpu(), hr_image.cpu()))

    return results

# Run testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DATSR().to(device)  # Assuming your model is initialized
results = test_model_with_pairs(model, test_loader, device)


from torchvision.utils import save_image

output_dir = "test/experment"
os.makedirs(output_dir, exist_ok=True)

for i, (output, hr_image) in enumerate(results):
    # Save the output image
    save_image(output[0], os.path.join(output_dir, f"output_{i}.png"))
