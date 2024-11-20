import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torchvision.transforms import Grayscale

# Grayscale Transformation
def grayscale_transform(image):
    return Grayscale(num_output_channels=1)(image)

# Feature Extractor (VGG19 Encoder)
class MatchingEncoder(nn.Module):
    def __init__(self):
        super(MatchingEncoder, self).__init__()
        self.encoder = vgg19(pretrained=True).features[:21]

    def forward(self, x):
        return self.encoder(x)

# Correspondence Matching
def correspondence_matching(lr_features, hr_features):
    # Implement matching logic (e.g., nearest neighbor or cross-correlation)
    # Returning dummy outputs for indices and scores
    return torch.rand_like(lr_features), torch.rand_like(hr_features)

# Texture Encoder
class TextureEncoder(nn.Module):
    def __init__(self):
        super(TextureEncoder, self).__init__()
        self.encoder = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.encoder(x)

# Domain Adaptation Module
class DomainAdaptation(nn.Module):
    def forward(self, lr_texture, hr_texture):
        # Simple example: average textures
        return (lr_texture + hr_texture) / 2

# Feature Aggregation
class FeatureAggregation(nn.Module):
    def forward(self, lr_features, matching_score, domain_features):
        return lr_features + matching_score + domain_features

# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.layers(x)

# Complete Model
class DATSR(nn.Module):
    def __init__(self):
        super(DATSR, self).__init__()
        self.matching_encoder = MatchingEncoder()
        self.texture_encoder = TextureEncoder()
        self.domain_adaptation = DomainAdaptation()
        self.feature_aggregation = FeatureAggregation()
        self.decoder = Decoder()

    def forward(self, lr_image, hr_image):
        # Pipeline 1
        lr_gray = grayscale_transform(lr_image)
        hr_gray = grayscale_transform(hr_image)
        lr_features = self.matching_encoder(lr_gray)
        hr_features = self.matching_encoder(hr_gray)
        matching_index, matching_score = correspondence_matching(lr_features, hr_features)

        # Pipeline 2
        lr_texture = self.texture_encoder(lr_image)
        hr_texture = self.texture_encoder(hr_image)
        domain_features = self.domain_adaptation(lr_texture, hr_texture)

        # Feature Aggregation and Decoding
        aggregated_features = self.feature_aggregation(lr_features, matching_score, domain_features)
        output = self.decoder(aggregated_features)
        return output


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for lr_image, hr_image in dataloader:
        lr_image, hr_image = lr_image.to(device), hr_image.to(device)
        optimizer.zero_grad()

        output = model(lr_image, hr_image)
        loss = criterion(output, hr_image)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def test(model, dataloader, device):
    model.eval()
    outputs = []

    with torch.no_grad():
        for lr_image, hr_image in dataloader:
            lr_image = lr_image.to(device)
            output = model(lr_image, hr_image)
            outputs.append(output.cpu())

    return outputs


from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize

def get_dataloader(root, batch_size, image_size):
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor()
    ])
    dataset = ImageFolder(root=root, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Initialize components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DATSR().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Prepare DataLoader
train_loader = get_dataloader("", batch_size=16, image_size=128)
test_loader = get_dataloader("../../RRSSRDset/RRSSRD/", batch_size=1, image_size=128)

# Training Loop
for epoch in range(10):  # Set epochs
    train_loss = train(model, train_loader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}, Loss: {train_loss}")


