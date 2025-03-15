# import os


# path = "/home/mahmoud-sayed/Desktop/Graduation Project/current/Data/OriginalData/ShapeNetVox32"
# counter = 0
# for dir in os.listdir(path):
#     folders = os.listdir(os.path.join(path, dir))
#     counter+=len(folders)
    
# kbs = (8*(len("1a74a83fa6d24b3cacd67ce2c72c02e"))*counter)/1024
# mbs = kbs/1024
# print(counter)
# print(kbs)
# print(mbs)
# ## storing the paths would require ~10.7 MBs which is inefficient considering that the training will be conducted on kaggle





# import os
# path = "/home/mahmoud-sayed/Desktop/Graduation Project/current/Data/OriginalData/ShapeNetVox32"

# print(sorted(os.listdir(path)))





# import os
# path_ = "/home/mahmoud-sayed/Desktop/Graduation Project/current/Data/Experimental/ShapeNetRendering/ShapeNetRendering"
# dirs = sorted(os.listdir(path_))

# for dir in dirs:
#     path = os.path.join(path_,dir)
#     files = os.listdir(path)
#     for file in files:
#         print('"'+file+'"'+",")
#     print("______")




# from utils.utils import visualize_from_file

# path = "/home/mahmoud-sayed/Desktop/Graduation Project/current/Data/OriginalData/ShapeNetVox32/02691156/10155655850468db78d106ce0a280f87/model.binvox"


# visualize_from_file(path)


# def اطبع(رءم):
#     print(رءم)


# تلاتة = 3
# اطبع(تلاتة)



# print([1]*10)


# import cv2

# cv2.cvtColor([], cv2.COLOR_BGR2RGB)



# import torchvision.transforms as T


# t = T.Compose([
#     T.Resize(224),

# ])


# import random
# lst1 = [1,2,3,4]
# lst2 = [4,3,2,1]

# both = list(zip(lst1, lst2))
# random.shuffle(both)
# lst1, lst2 = zip(*both)
# print(lst1)
# print(lst2)

# from torchvision.models import mobilenet_v3_small



import timm

model = timm.create_model("convnext_large_in22k", pretrained=True)
print(model)