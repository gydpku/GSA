import torch.nn as nn

#from models.resnet import ResNet18, ResNet34, ResNet50
#from models.resnet_imagenet import resnet18, resnet50
from CSL import tao as TL


def get_simclr_augmentation(P, image_size):

    # parameter for resizecrop
    #P.resize_fix = False
    resize_scale = (P.resize_factor, 1.0) # resize scaling factor,default [0.08,1]
   # if P.resize_fix: # if resize_fix is True, use same scale
    #    resize_scale = (P.resize_factor, P.resize_factor)

    # Align augmentation
    color_jitter = TL.ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
    color_gray = TL.RandomColorGrayLayer(p=0.2)
    resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=image_size)

    # Transform define #
    print("P",P.dataset)
    if P.dataset == 'imagenet': # Using RandomResizedCrop at PIL transform
        transform = nn.Sequential(
            color_jitter,
            color_gray,
        )
    elif P.dataset =='split_mnist':
        print("MNOSTYYY")
        transform = nn.Sequential(
              # 这个不会变换大小，但是会变化通道值，新旧混杂
             # 这个也不会，混搭
            resize_crop,  # 再次仿射取样，不会变大小
        )
    elif  P.dataset== "mnist":
        transform = nn.Sequential(
            # 这个不会变换大小，但是会变化通道值，新旧混杂
            # 这个也不会，混搭
            resize_crop,  # 再次仿射取样，不会变大小
        )

    elif P.dataset=="cifar10":
        transform = nn.Sequential(
            color_jitter,#这个不会变换大小，但是会变化通道值，新旧混杂
            color_gray,#这个也不会，混搭
            resize_crop,#再次仿射取样，不会变大小
        )


    return transform


def get_shift_module(P, eval=False):

    if P.shift_trans_type == 'rotation':
        shift_transform = TL.Rotation()
        K_shift = 4
    elif P.shift_trans_type == 'cutperm':
        shift_transform = TL.CutPerm()
        K_shift = 4
    else:
        shift_transform = nn.Identity()
        K_shift = 1#啥也不做，one_class=1

    if not eval and not ('sup' in P.mode):
        assert P.batch_size == int(128/K_shift)

    return shift_transform, K_shift


def get_shift_classifer(model, K_shift):

    model.shift_cls_layer = nn.Linear(model.last_dim, K_shift)#改成预测4类shift

    return model


def get_classifier(mode, n_classes=10):
    if mode == 'resnet18':
        classifier = ResNet18(num_classes=n_classes)
    elif mode == 'resnet34':
        classifier = ResNet34(num_classes=n_classes)
    elif mode == 'resnet50':
        classifier = ResNet50(num_classes=n_classes)
    elif mode == 'resnet18_imagenet':
        classifier = resnet18(num_classes=n_classes)
    elif mode == 'resnet50_imagenet':
        classifier = resnet50(num_classes=n_classes)
    else:
        raise NotImplementedError()

    return classifier

