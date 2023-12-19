import threading
import torch
import gc
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2, InterpolationMode
# import timm

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")

if str(device) == "cuda":
    torch.backends.cudnn.benchmark = True
print(device)

def calculate_mean_for_label(z_support, support_labels, label):
    # Select the elements of z_support corresponding to the current label and calculate their mean
    return z_support[torch.nonzero(support_labels == label)].mean(0)

def update(event: threading.Event=False, model=None, duplicate: int=1) -> None:
    BATCH_SIZE = 28

    image_size = 336
    normalize = v2.Normalize(mean = [0.48145466, 0.4578275, 0.40821073],
                                    std = [0.26862954, 0.26130258, 0.27577711])

    support_dataset = ImageFolder(
        root="./CatDataset/support",
        transform=v2.Compose(
            [   
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True), 
                v2.RandomEqualize(1.0),
                # v2.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
                # v2.RandomCrop([image_size, image_size]),
                v2.RandomHorizontalFlip(),
                # v2.RandomVerticalFlip(),
                v2.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
                v2.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 5.)),
                v2.Resize([image_size, image_size], interpolation=InterpolationMode.BICUBIC, antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                normalize
            ]
        )
    )

    support_loader = DataLoader(
        support_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        pin_memory=True
    )

    if len(support_loader) == 1:
        with torch.no_grad():
            for n in range(duplicate):
                support_images, support_labels = next(iter(support_loader))
                support_images = support_images.to(device, non_blocking=True)
                # Extract the features of support and query images
                z_support = model.forward(support_images)
                if n == 0:
                    final_z_support = z_support.detach().clone()
                    final_support_labels = support_labels.detach().clone()
                else:
                    final_z_support = torch.cat((final_z_support, z_support.detach().clone()))
                    final_support_labels = torch.cat((final_support_labels, support_labels.detach().clone()))
                del z_support, support_labels
                gc.collect()
                if device.type == "mps":
                    torch.mps.empty_cache()
                else:
                    torch.cuda.empty_cache()
            z_support = final_z_support
            # Infer the number of different classes from the labels of the support set
            n_way = len(torch.unique(final_support_labels))
    else:
        for n in range(duplicate):
            with torch.no_grad():
                for i, (support_images, support_labels) in enumerate(support_loader):
                    z_support = model.forward(support_images.to(device, non_blocking=True))
                    if i == 0 and n == 0:
                        final_z_support = z_support.detach().clone()
                        final_support_labels = support_labels.detach().clone()
                    else:
                        final_z_support = torch.cat((final_z_support, z_support.detach().clone()))
                        final_support_labels = torch.cat((final_support_labels, support_labels.detach().clone()))
                    del z_support, support_labels
                    gc.collect()
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    else:
                        torch.cuda.empty_cache()
        z_support = final_z_support
        print(torch.unique(final_support_labels))
        n_way = len(torch.unique(final_support_labels))
    print(z_support.shape, n_way)

    # Prototype i is the mean of all instances of features corresponding to labels == i
    # Run the calculation in parallel for each label
    parallel_results = Parallel(n_jobs=n_way)(delayed(calculate_mean_for_label)(z_support, final_support_labels, label) for label in range(n_way))
    z_proto = torch.cat(parallel_results)
    # z_proto = torch.cat(
    #     [z_support[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)]
    # )
    # Creating a reverse mapping to map tensor values to keys
    reverse_class_dict = {v: k for k, v in support_dataset.class_to_idx.items()}

    print(z_proto.shape, len(reverse_class_dict))
    torch.save(z_proto.clone(), "proto.pt")
    torch.save(reverse_class_dict, "class_dict.pt")
    if event:
        event.set()
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()
    # return z_proto, final_support_labels

# model = timm.create_model("eva02_small_patch14_336.mim_in22k_ft_in1k", pretrained=True, num_classes=0)
# model = timm.create_model("eva02_tiny_patch14_336.mim_in22k_ft_in1k", pretrained=True, num_classes=0)
# model = model.to(device)
# model.eval()

# update(False, model, 20)