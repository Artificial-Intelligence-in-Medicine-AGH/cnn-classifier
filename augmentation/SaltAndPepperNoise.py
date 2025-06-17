import random

import albumentations as A

class SaltAndPepperNoise(A.ImageOnlyTransform):
    def __init__(self, p=0.5, amount=0.02):
        super().__init__(p)
        self.amount = amount

    def apply(self, image, **params):
        h, w = image.shape

        for _ in range(10000):

            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            color = 0

            if image[y, x] < 125:
                color = 255

            image[y, x] = color

        return image