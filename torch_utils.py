import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image


from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
from io import open
import os
import pathlib
import glob
#import cv2
from torchvision import datasets, models, transforms



# load model

model = models.resnet18(pretrained=True,)

num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 3.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 3)


# save and load entire model

FILE = "best_checkpoint_tf.model"
#torch.save(model, FILE)
#model.load_state_dict(torch.load(FILE))
torch.load(FILE)
# loaded_model=torch.load(FILE)
model.eval()




#Transforms
transformer=transforms.Compose([
    transforms.Resize((150,150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])




# classes = {'txt':0, 'dia':1, 'tbl':2}

#prediction function
def get_prediction(img_path,transformer):
    classes = ['text','diagram','table']
    
    image=Image.open(img_path)
    
    image_tensor=transformer(image).float()
    
    
    image_tensor=image_tensor.unsqueeze_(0)
    
    if torch.cuda.is_available():
        image_tensor.cuda()
        
    input=Variable(image_tensor)
    
    
    output=model(input)
#     print(output)
#     print(output.shape)
    
    index=output.data.numpy().argmax()
    #print(index)
    
    pred=classes[index]
    

    return pred
    


