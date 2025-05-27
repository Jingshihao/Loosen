import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(  # nn.Sequential能够将一系列层结构组合成一个新的结构   features用于提取图像特征的结构
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            # input[3, 224, 224]  output[48, 55, 55] 48是卷积核个数 batch channel high weight
            # padding参数解释：如果是int型，比如说1 就是上下左右都补1列0  如果传入一个tuple(元组)的话 比如传入(1,2),1代表上下方各补一行0，2代表左右两侧各补两列0
            nn.ReLU(inplace=True),  # inplace这个参数可以理解为pytorch通过一种方法增加计算量，来降低内存使用容量的一种方法，可以通过这种方法在内存中载入更大的模型
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27] 没有设置stride是因为这个卷积层的步长是1，而默认的步长就是1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(  # 包含了三层全连接层 是一个分类器
            nn.Dropout(p=0.5),  # p是每一层随机失活的比例  默认是0.5
            nn.Linear(128 * 6 * 6, 2048),  # 第一个全连接层，全连接层的节点个数是2048个
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),  # 第二个全连接层
            nn.ReLU(inplace=True),
              # 第三个全连接层 输入的是数据集 类别的个数，默认是1000

        )
        self.fc = nn.Linear(2048, num_classes)
        if init_weights:  # 初始化权重
            self._initialize_weights()

    def forward(self, x):  # 正向传播过程
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # 进行一个展平处理  将传进来的变量x进行展平，从index=1 这个维度开始 也就是channel
        x = self.classifier(x)
        return x  # 得到网络的预测输出

    def _initialize_weights(self):
        for m in self.modules():  # 遍历每一层结构
            if isinstance(m, nn.Conv2d):  # 判断是哪一个类别
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # kaiming_normal初始化权重方法
                if m.bias is not None:  # 如果偏置不为空的话，那么就用0去初始化
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 如果这个实例是全连接层
                nn.init.normal_(m.weight, 0, 0.01)  # 采用normal，也就是正态分布来给我们的权重进行赋值，正态分布的均值等于0，方差是0.01
                nn.init.constant_(m.bias, 0)  # 将偏置初始化为0