import datasets
from torchvision.transforms import v2
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

num_images = 10

transformer = v2.Compose(
    [
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
    ]
)

root = os.path.join("datasets/", "WATERBIRDS", "processed")
train_data = datasets.VOCDetectParsed(
    root=root,
    image_set="train",
    transform=None,
    annotated_fraction=1,
    bbs_transform=transformer,
    plot=True,
)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=num_images,
    shuffle=True,
    num_workers=0,
    collate_fn=datasets.VOCDetectParsed.collate_fn,
)

inputs, classes, bb_box_list, og_inputs, og_classes, og_bb_box_list = next(
    iter(train_loader)
)

fig, axs = plt.subplots(num_images, 2, figsize=(5, 4 * num_images))
for i in range(num_images):
    image = inputs[i]
    bbs = bb_box_list[i]

    og_image = og_inputs[i]
    og_bbs = og_bb_box_list[i]

    axs[i][1].imshow(torch.movedim(image[:3, :, :], 0, -1))
    axs[i][1].set_title("Transformed image + bb box")

    axs[i][0].imshow(torch.movedim(og_image[:3, :, :], 0, -1))
    axs[i][0].set_title("Original image + bb box")

    for bb in bbs:
        cl, xmin, ymin, xmax, ymax = bb
        axs[i][1].add_patch(
            patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fc="none",
                ec="#00FFFF",
                lw=5,
            )
        )

    for bb in og_bbs:
        cl, xmin, ymin, xmax, ymax = bb
        axs[i][0].add_patch(
            patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fc="none",
                ec="#00FFFF",
                lw=5,
            )
        )

plt.tight_layout()
plt.savefig("./images/transform_examples.png")
