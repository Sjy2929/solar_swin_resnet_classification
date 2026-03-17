1.安装依赖
pip install -r requirements.txt

2. 训练模型
冻结骨干网络训练（适合小数据集）：
python train.py --freeze_backbone
全部参数训练
python train.py
可选参数
--data_dir: 数据集路径 (default: ./solar_dataset_split)
--batch_size: 批大小 (default: 32)
--num_epochs: 训练轮数 (default: 260)
--lr: 初始学习率 (default: 1e-3)
--freeze_backbone: 冻结骨干网络权重
--swin_type: Swin Transformer类型 (default: tiny)
--save_dir：训练模型路径(default: ./saved_models)
--warmup_epochs：(default=10, help='Warmup epochs')
--min_lr：(default=1e-6, help='Minimum learning rate')
--warmup_lr：(default=1e-4, help='Warmup starting learning rate')

3.评估模型
python evaluate.py
可选参数
--data_dir: 数据集路径 (default: .solar_dataset_split)
--model_path: 模型路径 (default: best_model.pth)
--batch_size: 批大小 (default: 32)
--num_classes: 类别数量 (default: 5)

4.单图像预测
python predict.py --image_path path/to/your/image.jpg
可选参数
--image_path: 输入图像路径 (required)
--model_path: 模型路径 (default: best_model.pth)
--num_classes: 类别数量 (default: 5)

5.模型特点
双骨干架构：结合Swin Transformer的全局建模能力和ResNet的局部特征提取
特征融合：通过1×1卷积融合两种特征
迁移学习：使用ImageNet预训练权重初始化
冻结选项：可选择冻结骨干网络权重，只训练分类头
分层学习率：不同模块使用不同学习率

6.数据集来源：
https://www.kaggle.com/datasets/gitenavnath/solar-augmented-dataset/data
该数据集包含较为全面的太阳能电池板图像集合，并分为六个标记类别：干净、积尘、鸟粪、积雪、电气故障和物理故障。
