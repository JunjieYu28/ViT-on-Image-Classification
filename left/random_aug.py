#Code from https://github.com/DeepVoltaire/AutoAugment

import numpy as np
from .ops import *
from torchvision.transforms.functional import to_pil_image, to_tensor

class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.0, "invert", 9, 0.0, "contrast", 4, fillcolor)
            # SubPolicy(0.4, "rotate", 4, 0.4, "translateX", 7, fillcolor),
            # SubPolicy(0.4, "sharpness", 4, 0.4, "sharpness", 6, fillcolor),
            # SubPolicy(0.4, "shearY", 4, 0.4, "translateY", 5, fillcolor),

            # SubPolicy(0.4, "shearY", 4, 0.4, "posterize", 6, fillcolor),
            # SubPolicy(0.4, "color", 3, 0.4, "brightness", 6, fillcolor),
            # SubPolicy(0.4, "sharpness", 6, 0.4, "brightness", 6, fillcolor),
            # SubPolicy(0.4, "equalize", 5, 0.4, "equalize", 4, fillcolor),
            # SubPolicy(0.4, "contrast", 7, 0.4, "sharpness", 5, fillcolor),

            # SubPolicy(0.4, "color", 5, 0.4, "translateX", 8, fillcolor),
            # SubPolicy(0.4, "translateY", 3, 0.4, "sharpness", 6, fillcolor),
            # SubPolicy(0.4, "brightness", 6, 0.4, "color", 5, fillcolor),
            # SubPolicy(0.4, "solarize", 2, 0.4, "invert", 3, fillcolor),

            # SubPolicy(0.4, "equalize", 5, 0.4, "equalize", 4, fillcolor),
            # SubPolicy(0.4, "color", 6, 0.4, "equalize", 3, fillcolor),
            # SubPolicy(0.4, "brightness", 3, 0.4, "color", 5, fillcolor),

            # SubPolicy(0.4, "translateY", 4, 0.4, "translateY", 5, fillcolor),
            # SubPolicy(0.4, "equalize", 6, 0.4, "invert", 3, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        img = to_pil_image(img)
        return to_tensor(self.policies[policy_idx](img))

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"

class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int32),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        func = {
            "shearX": ShearX(fillcolor=fillcolor),
            "shearY": ShearY(fillcolor=fillcolor),
            "translateX": TranslateX(fillcolor=fillcolor),
            "translateY": TranslateY(fillcolor=fillcolor),
            "rotate": Rotate(),
            "color": Color(),
            "posterize": Posterize(),
            "solarize": Solarize(),
            "contrast": Contrast(),
            "sharpness": Sharpness(),
            "brightness": Brightness(),
            "autocontrast": AutoContrast(),
            "equalize": Equalize(),
            "invert": Invert()
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img