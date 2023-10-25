import matplotlib.pyplot as plt

def show_images(*imgs):
    figsize = (30, 30 * len(imgs))

    plt.figure(figsize=figsize)

    for i, img in enumerate(imgs):
        plt.subplot(i)
        plt.imshow(img.cpu().data.numpy().transpose(1, 2, 0))
        plt.xticks([])
        plt.yticks([])
    
    plt.show()