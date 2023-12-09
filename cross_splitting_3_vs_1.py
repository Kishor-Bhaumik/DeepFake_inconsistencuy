
import random, natsort,json, glob
f2f = '/media/data1/FaceForensics_Dec2020/FaceForensics++/v1face_data/data_raw/manipulated_sequences/Face2Face/raw/v1faces/*'
FS = '/media/data1/FaceForensics_Dec2020/FaceForensics++/v1face_data/data_raw/manipulated_sequences/FaceSwap/raw/v1faces/*'
NT = '/media/data1/FaceForensics_Dec2020/FaceForensics++/v1face_data/data_raw/manipulated_sequences/NeuralTextures/raw/v1faces/*'
DF ='/media/data1/FaceForensics_Dec2020/FaceForensics++/v1face_data/data_raw/manipulated_sequences/Deepfakes/raw/v1faces/*'

org_dir = '/media/data1/FaceForensics_Dec2020/FaceForensics++/v1face_data/data_raw/original_sequences/youtube/raw/v1faces/*' 
image_to_be_taken_in_a_frame = 80
img_per_frame = 8

'''
lets say each f2f , FS, NT, Df has 1000 folders. Lets say each folder contains 200 images. This 200 images are from a video. the name of the images 
are 1.png, 2,png, ....200.png.  so if we sort this name , then it will be a video frame. we take first 80 images  (1.png to 80.png ) per video.

then  we split it into 10 , so each division has 8 images. 
We want to build a dataset where we will train on 3 forgery types (lets say f2f,FS,NT) and test on 1 forgery type(lets say DF)

But if we take all the dataset from 3 forgeries , this will create a class imbalance problem. since the real data will be smaller.

So for training we take:
first 800 folders from real, random 266 folders from f2f, random 266 folders from FS and random 266 folders from NT 

for testing we take:
last 200 folders from real and randon 200 folders form DF. 
'''


def read_image(dir,label, mode):
    files=glob.glob(dir) 
    label = 0 if label=="Fake" else 1
    if mode =='train':
        if label == 0:
            ind= random.randint(0, 700)
            files = files[ind: ind+266]
        else:
            files =files[:800]
    
    else:
        if label ==0:
            ind= random.randint(0, 790)
            files = files[ind: ind+200]
        
        else:
            files = files[:200]

    
    taken_images_with_label= []

    for File in files:
        images= glob.glob(File+"/*")
        image_name_sorted = natsort.natsorted(images)
        images_=  image_name_sorted[:image_to_be_taken_in_a_frame]

        for r in range(0,len(images_)-img_per_frame+1, img_per_frame):
            taken_images = images_[r:r+img_per_frame]
            taken_images_with_label.append(taken_images+[label])


    return taken_images_with_label

training = [ f2f,FS ,NT]
testing = [DF]
all_train=[]



for Dir in training :
    imgs= read_image(Dir, label = 'Fake', mode = 'train')
    all_train+= imgs

all_train+= read_image(org_dir, label = 'real', mode = 'train')

random.shuffle(all_train)

all_test=[]
for Dir in testing:
    imgs = read_image(Dir, label = 'Fake', mode = 'test')
    all_test+= imgs

all_test+= read_image(org_dir, label = 'real', mode = 'test')

random.shuffle(all_test)

dataset_paths = dict({'train': all_train ,'test': all_test})

json_object = json.dumps(dataset_paths , indent=3)
 
with open("DeepFake_test.json", "w") as outfile:
    outfile.write(json_object)




# f = open('face2face_test.json')
# data = json.load(f)

# test = data['test']


