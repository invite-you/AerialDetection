import imgaug as ia
import imgaug.augmenters as iaa

class CustomAugmentation(object):

    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential([
                        iaa.Fliplr(0.5),
                        iaa.Flipud(0.5),
                        iaa.Affine(scale=(0.5, 1.5), rotate=(-380, 380),
                                translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},
                                mode=['symmetric', 'reflect'], cval=(0, 255)),
                        iaa.GammaContrast((0, 2.0)),
                        sometimes(iaa.SomeOf(1, [iaa.MultiplyAndAddToBrightness(mul=(0.5,
                                1.5), add=(-10, 10)), iaa.Grayscale(alpha=(0.0, 1.0))])),
                        sometimes(iaa.SomeOf(1, [iaa.GaussianBlur(sigma=(0.2, 2.0)),
                                iaa.AverageBlur(k=(1, 5)), iaa.MedianBlur(k=(1, 5)),
                                iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250),
                                sigma_space=(10, 250))])),
                        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.15))),
                    ])

    def __call__(self, img, boxes, masks, labels, filename):
        bboxes = []
        for box in boxes[:, :4]:
            x1, y1, x2, y2 = box
            bboxes.append( ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2) )
        print("####")
        print(img.shape, bboxes.shape, masks.shape)
        image_aug, bbs_aug, segmaps_aug = self.seq(images=[image], bounding_boxes=[bboxes], segmentation_maps=[masks])
        image_aug, bbs_aug, segmaps_aug = image_aug[0], bbs_aug[0], segmaps_aug=[0]
        print(image_aug.shape, bbs_aug.shape, segmaps_aug.shape)
        
        return image_aug, bbs_aug, segmaps_aug, labels
