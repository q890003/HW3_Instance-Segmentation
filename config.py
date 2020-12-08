##################################################
# Default Config
##################################################
# Directory
ckpt_dir = "./checkpoints/"  # saving directory of .ckpt models
result_pth = "./results/"
directories = [ckpt_dir, result_pth]

# Logging Config
logg_path = "./log/"

# Dataset/Path Config
train_folder = "/home/mbl/Yiyuan/CV_hw3/data/train_images/"  # training dataset path
test_folder = "/home/mbl/Yiyuan/CV_hw3/data/test_images/"  # testing images
annotation_file = "./data/pascal_train.json"  # training label

split = 0.1  # percentage of validation set
workers = 4  # number of Dataloader workers

# Detection Task Config
num_classes = 21  # 1 class (person) + background

##################################################
# Training Config
##################################################
device = "cuda:0"
model_name = "maskRCNN_"
log_name = "train.log"  # Beta


# Hyper-parameters Config
epochs = 15  # number of epochs
batch_size = 4  # batch size
learning_rate = 1e-3  # initial learning rate

##################################################
# Eval Config
##################################################
prediction_file_name = model_name + ".json"  # saving prediction result
