from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import pdb
import time
import torch

#pdb.set_trace()

img = cv2.imread('testframe.jpg', cv2.IMREAD_COLOR)
img = img[...,::-1]


precision= torch.float16

sam = sam_model_registry["default"](checkpoint = './checkpoints/sam_vit_h_4b8939.pth', dtype=precision).cuda().to(precision)

mask_generator = SamAutomaticMaskGenerator(sam, dtype=precision)

stime=time.time()

for i in range(10):
	masks = mask_generator.generate(img)

etime=time.time()

print(f"Took {etime - stime :.3f} seconds for 10 inferences")
