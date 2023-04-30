import numpy as np
import cv2
import json
import pycocotools.mask as maskutils
import pdb
import time
import os


def visualize(imgpath, labelpath):
	pass
	img=cv2.imread(imgpath, 1)
	with open(labelpath, 'r') as f:
		anndata=json.load(f)

	h, w = anndata['image']['height'], anndata['image']['width']

	all_rles=[]
	masks=[]

	for ann in anndata['annotations']:
		segm = ann['segmentation']
		if type(segm)==list:
			rles = maskutils.frPyObjects(segm, h, w)
			rle = maskutils.merge(rles)
			all_rles.append(rle)
		elif type(segm['counts']) == list:
			rle=maskutils.frPyObjects(segm, h, w)
			all_rles.append(rle)
		else:
			rle = segm
			all_rles.append(rle)

	np.random.shuffle(all_rles)
	nrles=len(all_rles)
	csize=nrles//3

	for idx in range(3):
		start=idx*csize
		end= nrles if idx==2 else (idx+1)*csize
		rles=all_rles[start:end]
		rle = maskutils.merge(rles)
		this_mask=maskutils.decode(rle)
		masks.append(this_mask)


	colors=np.array([[255,0,0], [0,255,0], [0,0,255]], dtype=np.uint8)
	
	overlay = np.zeros_like(img)

	for idx, mask in enumerate(masks):
		overlay[mask>0,:] = colors[idx%3]

	img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

	cv2.imshow('masks', img)
	k=cv2.waitKey(1)
	if k==ord('q'):
		quit()
	time.sleep(2)


if __name__ == '__main__':
	images=[f'./data/images/{f}' for f in os.listdir('./data/images')]
	#jsons =[f'./data/labels/{f}' for f in os.listdir('./data/labels')]

	np.random.shuffle(images)
	#jsons=sorted(jsons)

	for imgpath in (images):
		#assert img[img.rfind('.')]
		stem = imgpath[imgpath.rfind('/')+1:imgpath.rfind('.')]
		annpath = f'./data/labels/{stem}.json'
		visualize(imgpath, annpath)
