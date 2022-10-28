import torch
import sys, os, math
from PIL import Image
import matplotlib.pyplot as plt


from netdissect import (
    nethook,
    imgviz,
    # show,
    segmenter,
    renormalize,
    upsample,
    tally,
    pbar,
    setting,
)

# Import packates
torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)  # not training anything

# Load some data
ds = setting.load_dataset("places", "val")
iv = imgviz.ImageVisualizer(224, source=ds, percent_level=0.99)
# iv.image(ds[0][0]).show()

# Load a pretrained classifier model
model = setting.load_vgg16()
model = nethook.InstrumentedModel(model)
# model.cuda()
renorm = renormalize.renormalizer(source=ds, target="zc")
ivsmall = imgviz.ImageVisualizer((56, 56), source=ds, percent_level=0.99)

target_class = ds.classes.index("soccer_field")
print(target_class)

indexes = range(100, 112)
batch = torch.stack([ds[i][0] for i in indexes])
_, preds = model(batch).max(1)

layername = "features.conv5_3"
model.retain_layer(layername)
# model(batch.cuda())
model(batch.cpu())
acts = model.retained_layer(layername).cpu()
print(acts.shape)

images_list = [
    [
        [ivsmall.masked_image(batch[imagenum], acts[imagenum], unitnum)],
        [ivsmall.heatmap(acts[imagenum], unitnum, mode="nearest")],
        "unit %d" % unitnum,
    ]
    for unitnum in range(acts.shape[1])
    for imagenum in [6]
]


def combine_images(images):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


images_list_new = [[i[0], j[0]] for i, j, _ in images_list]
images_title = [i for _, _, i in images_list]

images = [combine_images(i) for i in images_list_new]


def show(images, columns, titles=None, if_save=False, filename="test"):
    num = len(images)
    rows = math.ceil(num / columns)

    fig = plt.figure(figsize=(columns * 2, rows))
    # fig.tight_layout(pad=0.1)

    ax = []
    for i in range(num):
        img = images[i]
        ax.append(fig.add_subplot(rows, columns, i + 1))

        if titles is not None:
            ax[-1].set_title(titles[i], pad=0)
            ax[-1].axis("off")
        plt.imshow(img)
    plt.subplots_adjust(top=1.1)
    if if_save:
        plt.savefig(f"{filename}.svg", bbox_inches="tight")
    plt.show()


show(images, 5, images_title)
