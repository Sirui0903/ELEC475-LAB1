import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import model


batch_size = 1


def interpolate_and_plot(steps, model, image1, image2):
    model.eval()
    with torch.no_grad():
        #       encode the images
        bottleneck1 = model.encode(image1.view(1, -1))
        bottleneck2 = model.encode(image2.view(1, -1))

        #       generate zeros array
        interpolations = torch.zeros((steps, *bottleneck1.size()))

        for i in range(steps):
            interpolations[i] = ((i / (steps - 1)) * bottleneck1 + (1 - (i / (steps - 1))) * bottleneck2)

        plt.figure(figsize=(15, 5))
        plt.axis('off')

        #   plot images
        for i in range(steps):
            restructed_image = model.decode(interpolations[i])
            output_image = restructed_image.view(1, 28, 28).cpu().numpy()

            plt.subplot(1, steps, i + 1)
            plt.imshow(output_image.squeeze(), cmap='gray')

        plt.show()
    return interpolations


def addNoise(image, noise_factor=0.2):
    noisy = image + noise_factor * torch.randn_like(image)
    return noisy


def evaluate_step5(model, loader):
    print('evaluating ...')
    model.eval()

    with torch.no_grad():
        data, _ = next(iter(loader))

        #   put the image into the addNoisy function and flatten it
        noisy_data = addNoise(data)
        noisy_image = noisy_data.view(28,28).numpy()
        noisy_data = noisy_data.view(noisy_data.size(0), -1)
        reconstructed_data = model(noisy_data)
        reconstructed_image = reconstructed_data.view(28,28).numpy()

        #   plot three graphs
        f = plt.figure()
        f.add_subplot(1, 3, 1)
        plt.imshow(data.view(28,28).numpy(), cmap='gray')
        plt.title('Original Image')
        f.add_subplot(1, 3, 2)
        plt.imshow(noisy_image, cmap='gray')
        plt.title('Denoised')
        f.add_subplot(1, 3, 3)
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title('Noisy Image')

        plt.show()


if __name__ == '__main__':
    #   read command
    parser1 = argparse.ArgumentParser()
    parser1.add_argument('-l', type=str, default='MLP.8.pth')
    args = parser1.parse_args()

    #   initialize the model
    model = model.autoencoderMLP4Layer()
    model.load_state_dict(torch.load(args.l))

    transform = transforms.Compose([transforms.ToTensor()])
    eval_set = MNIST('./data/mnist', train=False, download=True, transform=transform)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True)
    evaluate_step5(model, eval_loader)

    # actual image to tensor
    image1, _ = next(iter(eval_loader))
    image2, _ = next(iter(eval_loader))

    # Number of interpolation steps
    steps = 10
    interpolate_and_plot(steps, model, image1, image2)





