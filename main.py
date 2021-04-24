import torch.optim as optim, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train.trainer import DCGANTrainer
from train.model import Generator, Discriminator

def main():
    device = 'cuda:0'

    # Build trainer
    z_dim = 62
    genarator = Generator(z_dim=z_dim)
    discriminator = Discriminator()
    trainer = DCGANTrainer(genarator, discriminator, z_dim, device)

    # Settings
    num_epochs = 50
    batch_size = 128
    g_optimizer = optim.Adam(genarator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    loss_fn = nn.CrossEntropyLoss()

    # Download MNIST and make dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train DCGAN
    trainer.train_model(data_loader, loss_fn, g_optimizer, d_optimizer, batch_size, num_epochs)

    # Generate images
    trainer.generate_image(image_num=64, file_name='test.png')


if __name__ == "__main__":
    main()