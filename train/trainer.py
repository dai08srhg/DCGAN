from train.model import Generator, Discriminator
import torch
from torchvision.utils import save_image


class DCGANTrainer():
    def __init__(self, generator: Generator, discriminator: Discriminator, z_dim, device, log_dir='./logs'):
        self.generator = generator
        self.discriminator = discriminator
        self.z_dim = z_dim
        self.device = device
        self.log_dir = log_dir

    def train_model(self, data_loader, loss_fn, g_optimizer, d_optimizer, batch_size, epochs, log_epochs=[1,10,25,50]):
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        # 本物のラベルは1
        y_reals = torch.ones((batch_size), dtype=torch.int64).to(self.device)
        # 偽物のラベルは0
        y_fakes = torch.zeros((batch_size), dtype=torch.int64).to(self.device)
        for epoch in range(epochs):
            self.generator.train()
            self.discriminator.train()
            for real_images, _ in data_loader:
                z = torch.rand((batch_size, self.z_dim)).to(self.device)  # 乱数生成
                real_images = real_images.to(self.device)  # real images
                fake_images = self.generator(z)  # fake images生成

                # discriminato(識別器の学習)
                d_optimizer.zero_grad()
                # realとfakeを識別
                d_reals = self.discriminator(real_images)
                d_fakes = self.discriminator(fake_images.detach())  # fake_imagesを通して勾配がgeneratorに伝わらないようにdetach()
                # loss計算
                d_real_loss = loss_fn(d_reals, y_reals)
                d_fake_loss = loss_fn(d_fakes, y_fakes)
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                d_optimizer.step()  # detach()してるためgeneratorの更新はない

                # generator(生成器の学習)
                g_optimizer.zero_grad()
                fake_images = self.generator(z)  # fake images生成
                d_fakes = self.discriminator(fake_images)
                g_loss = loss_fn(d_fakes, y_reals)  # fakeとrealの誤差を最小化する
                g_loss.backward()
                g_optimizer.step()
            
            if epoch+1 in log_epochs:
                self.generate_image(64, file_name=f'{epoch+1}.png')

    def generate_image(self, image_num, file_name):
        sample_z = torch.rand((image_num, self.z_dim)).to(self.device)
        samples = self.generator(sample_z).to('cpu')
        self.save_images(samples, file_name)

    def save_images(self, images, file_name):
        save_image(images, f'{self.log_dir}/{file_name}')


