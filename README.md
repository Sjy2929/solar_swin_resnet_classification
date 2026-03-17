1.课程项目：本项目基于太阳能电池板的图像集合，通过深度学习的模型训练和评估，对运维过程中电池板组件的问题故障进行分类预测。

2.数据集来源：
https://www.kaggle.com/datasets/gitenavnath/solar-augmented-dataset/data
该数据集包含较为全面的太阳能电池板图像集合，并分为六个标记类别：干净、积尘、鸟粪、积雪、电气故障和物理故障。

3.项目介绍
项目所采用的模型是一个混合架构，结合了 ResNet50 卷积神经网络和 Swin Transformer 模型的优势，用于图像分类任务。它通过将两种不同类型模型提取的特征进行融合，从而能更全面地捕捉图像的特征信息，从而提升分类性能。

4.预训练权重文件win_tiny_patch4_window7_224.pth放在model文件下，不可随意更改位置。

5.代码具体运行

5.1.安装依赖
pip install -r requirements.txt

5.2 训练、评估和预测代码操作见run_code.txt
