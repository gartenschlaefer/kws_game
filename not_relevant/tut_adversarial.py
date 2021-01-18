"""
adversarial nn tutorial
adapted form: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from IPython.display import HTML, display





class Generator(nn.Module):
    def __init__(self, ngpu):

        super(Generator, self).__init__()

        self.ngpu = ngpu

        self.main = nn.Sequential(

            # input is Z, going into a convolution
            #nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            #nn.BatchNorm2d(ngf * 8),

            # z
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
            #nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 4),

            # 4ngf x 4 x 4
            nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # state size. (ngf*4) x 8 x 8
            #nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 2),

            # 2ngf x 8 x 8
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # state size. (ngf*2) x 16 x 16
            #nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf),
            #nn.ReLU(True),


            # state size. (ngf) x 32 x 32
            #nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            # state size. (nc) x 64 x 64

            # ngf x 16 x 16
            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False),

            # nc x 32 x 32
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)




class Discriminator(nn.Module):

    def __init__(self, ngpu):

        super(Discriminator, self).__init__()

        self.ngpu = ngpu

        self.main = nn.Sequential(

            # nc x 32 x 32
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf x 16 x 16
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 2ndf x 8 x 8
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 4ndf x 4 x 4
            nn.Conv2d(in_channels=ndf * 4, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)



def weights_init(m):
    """
    custom weights initialization called on netG and netD
    """

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def print_loss(G_losses, D_losses):
    """
    print loss functions
    """

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def show_some_images(trainloader):
    """
    show some images from dataset
    """

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(next(iter(trainloader))[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()



if __name__ == '__main__':

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 128

    # Number of channels in the training images. For color images this is 3
    nc = 3

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    #ngf = 64
    ngf = 32

    # Size of feature maps in discriminator
    #ndf = 64
    ndf = 32

    # Number of training epochs
    num_epochs = 2

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1


    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


    # --
    # cifar

    # transform init
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # training and test set
    trainset = torchvision.datasets.CIFAR10(root='./ignore/cifar10', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./ignore/cifar10', train=False, download=True, transform=transform)

    # loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # save parameter path
    param_path = './ignore/cifar10/cifar_net.pth'


    # show some images
    #show_some_images(trainloader)



    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the model
    print(netG)


    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)


    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")

    # For each epoch
    for epoch in range(num_epochs):

        # For each batch in the dataloader
        for i, data in enumerate(trainloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            # --
            # Train with all-real batch

            # zero gradients
            netD.zero_grad()

            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)

            # create labels for real data
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            
            #print("data: ", data[0].shape)
            #print("real_cpu: ", real_cpu.shape)
            #print("output: ", output)
            #print("output: ", output.shape)
            #print("label: ", label.shape)

            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()



            # --
            # Train with all-fake batch

            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            #print("noise: ", noise.shape)


            # Generate fake image batch with G
            fake = netG(noise)
            #print("fake: ", fake.shape)

            label.fill_(fake_label)

            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            #print("output: ", output.shape)

            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)

            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()


            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################

            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)

            # Calculate G's loss based on this output
            errG = criterion(output, label)

            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()

            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(trainloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(trainloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1



    # --
    # prints

    print_loss(G_losses, D_losses)


    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]

    # animation
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    # test
    #display(HTML(ani.to_jshtml()))

    # show ani
    plt.show()

    # save
    ani.save("./ignore/out_ep-{}_ndf-{}.mp4".format(num_epochs, ndf))
