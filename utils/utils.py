import matplotlib.pyplot as plt
import cv2


def plot_img_and_mask(img, mask,final):
    # classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, 3)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[1].set_title('Model output')
    ax[1].imshow(mask)
    ax[2].set_title('Final output')
    ax[2].imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    # if classes > 1:
    #     for i in range(classes):
    #         ax[i + 1].set_title(f'Output mask (class {i + 1})')
    #         ax[i + 1].imshow(mask[i, :, :])
    # else:
    #     ax[1].set_title(f'Output mask')
    #     ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()
