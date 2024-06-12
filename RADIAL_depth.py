import cv2
from transformers import pipeline
from PIL import Image
import requests
import torch 
import os
import sys
sys.path.append("C:\\Users\\nxg05733\\Depth-Anything")
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from multiprocessing import Pool, freeze_support

model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14").to('cuda').eval()

transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

def get_depth_maps(folder, output_folder, root_folder ):
    current_folder = os.path.join(root_folder, folder)
    images = os.listdir(current_folder)
    if not os.path.exists(os.path.join(output_folder, folder)):
        os.makedirs(os.path.join(output_folder, folder))
    for i,image in enumerate(images):
      image_file = Image.open(os.path.join(current_folder, image))
      image_file = np.array(image_file) /255.0

      h, w = image_file.shape[:2]
      image_file = transform({'image' : image_file})['image']
      image_file = torch.from_numpy(image_file).unsqueeze(0).to('cuda')

      with torch.no_grad():
          depth = model(image_file)
          depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
          depth = depth.cpu().numpy()
          file_name = os.path.join(output_folder,folder,f"Depth_{i}.npy")
          np.save(file_name,depth)
      

if __name__ == '__main__':
  freeze_support()
  output_folder = "depth_maps"
  root_folder = "C:\\Users\\nxg05733\\RADIal\\resized_2\\images"
  folders = os.listdir(root_folder)
  folders_to_process = [(folder, output_folder, root_folder) for folder in folders]
  print("Starting")
  with Pool(processes=6) as pool:
      pool.starmap(get_depth_maps, folders_to_process)
  pool.close()
  pool.join()
  print("Completed ")

    