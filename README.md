# project2_CNN
# ResNet을 사용한 강아지 종류를 구분하는 알고리즘
# CNN vs FCN(fully connected layer)

# 실행방법
# Basic model
python train.py --train_dir ./images/images --test_dir ./images/images  
  
# FCN
python train.py --train_dir ./images/images --test_dir ./images/images --device cpu --arch resfcn18
  
# CNN
python train.py --train_dir ./images/images --test_dir ./images/images --device cpu --arch resnet18  

# data
kaggle : https://www.kaggle.com/yaswanthgali/dog-images
