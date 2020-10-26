import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

import numpy as np
import cv2

from PIL import Image, ImageDraw

class CustomAugmentation(object):

    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(0.3, aug)
        self.seq = iaa.Sequential([
                        iaa.Sometimes(iaa.CropToFixedSize(width=640, height=640)),
                        iaa.Fliplr(0.5),
                        iaa.Flipud(0.5),
                        iaa.Affine(rotate=(-380, 380),
                                translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},
                                #mode=['symmetric', 'reflect'], # bbox는 reflect 되지 않음
                                cval=(0, 0)),                             
                        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.11))),
                    ])

    def __call__(self, img, boxes, masks, labels, filename):
        #print("####")
        #print(type(img), type(boxes), type(masks[0]), len(masks), np.unique(masks[0]))
        #print(img.shape, boxes.shape, masks[0].shape)
        bboxes = []
        for box in boxes[:, :4]:
            x1, y1, x2, y2 = box
            bboxes.append( ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2) )        
        
        # mask를 위해서 polygon 만들기
        points = []
        for gt_mask in masks:
            contours, ct = cv2.findContours(gt_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                # 각도 있는 다각형                
                rc = cv2.minAreaRect(contours[0])
                points.append( cv2.boxPoints(rc) )
                #cv2.dravwContours(im, [box], 0, (0,255,0),3)
                # 각도 없는 사각형
                #rect = cv2.boundingRect(c)
                #x,y,w,h = rect
                #cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                #cv2.putText(im,'Moth Detected',(x+w+10,y+h),0,0.3,(0,255,0))
                
        mulpoly = [ Polygon(point) for point in points]
        psoi = ia.PolygonsOnImage(mulpoly, shape=img.shape)


        image_aug, bbs_aug, poly_aug = self.seq(images=[img], bounding_boxes=[bboxes], polygons=[psoi])
        image_aug, bbs_aug, poly_aug = image_aug[0], bbs_aug[0], poly_aug[0]

        # ia.BoundingBox 된 boxes 원래 형태로 돌려놓기        
        bbs = []
        for bb in bbs_aug:
            bbs.append(bb.coords.reshape(4))
        bbs_aug = np.array(bbs)

        # polygon이 된 mask를 원래 형태로 돌려놓기        
        masks_aug = []
        for poly in poly_aug:
            img = Image.new('L', (image_aug.shape[0], image_aug.shape[0]), 0)
            ImageDraw.Draw(img).polygon(poly.coords, outline=1, fill=1)
            masks_aug.append(np.array(img))

        #print(type(image_aug), type(bbs_aug), type(masks_aug[0]), len(masks_aug), np.unique(masks_aug[0]))
        #print(image_aug.shape, bbs_aug.shape, len(masks_aug), masks_aug[0].shape)
        
        return image_aug, bbs_aug, masks_aug, labels
