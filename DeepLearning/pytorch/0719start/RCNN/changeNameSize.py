import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt


trans = transforms.Compose([
    # transforms.Resize((64,128))

])

train_data = torchvision.datasets.ImageFolder(root='H://Resources//AIschool//DeepLearning//dataSet//archive//train', transform=None)
test_data = torchvision.datasets.ImageFolder(root='H://Resources//AIschool//DeepLearning//dataSet//archive//test', transform=None)

# for num, value in enumerate(train_data):
#     data, label = value
#     print(num, data, label)
#
#     if (label == 0):
#         data.save(f'H://Resources//AIschool//DeepLearning//dataSet//archive//rename//train//paper//paper_{num}.png')
#     elif label == 1:
#         rock_num = num-712
#         data.save(f'H://Resources//AIschool//DeepLearning//dataSet//archive//rename//train//rock//rock_{rock_num}.png')
#     elif label == 2:
#         scissors_num = num-1438
#         data.save(f'H://Resources//AIschool//DeepLearning//dataSet//archive//rename//train//scissors//scissors_{scissors_num}.png')

for num, value in enumerate(test_data):
    data, label = value
    print(num, data, label)

    if (label == 0):
        data.save(f'H://Resources//AIschool//DeepLearning//dataSet//archive//rename//test//paper//paper_{num}.png')
    elif label == 1:
        # rock_num = num-712
        data.save(f'H://Resources//AIschool//DeepLearning//dataSet//archive//rename//test//rock//rock_{num}.png')
    elif label == 2:
        # scissors_num = num-1438
        data.save(f'H://Resources//AIschool//DeepLearning//dataSet//archive//rename//test//scissors//scissors_{num}.png')