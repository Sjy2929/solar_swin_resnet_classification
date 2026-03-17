import os
import splitfolders

def split_dataset(raw_data_dir, output_dir, ratios=(0.7, 0.15, 0.15)):
    """
    划分数据集为训练集、验证集和测试集
    
    参数:
    raw_data_dir: 原始数据集路径
    output_dir: 输出目录
    ratios: 训练、验证、测试集的比例
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用splitfolders划分数据集
    splitfolders.ratio(
        raw_data_dir,
        output=output_dir,
        seed=42,
        ratio=ratios,
        group_prefix=None,
        move=False
    )
    
    print(f"数据集已划分到: {output_dir}")
    print(f"训练集: {ratios[0]*100}%, 验证集: {ratios[1]*100}%, 测试集: {ratios[2]*100}%")

if __name__ == "__main__":
    # 设置路径
    raw_data_dir = "swin_resnet_classification/data/PRoject"  # 原始数据路径
    output_dir = "swin_resnet_classification/data/solar_dataset_split"  # 划分后的数据集路径
    
    # 划分数据集
    split_dataset(raw_data_dir, output_dir)