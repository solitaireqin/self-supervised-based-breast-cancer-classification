import os
from pretrain.main import args

def get_indices():
    indices = {}
    person = 1  # 病人编号从 1 开始
    mid = []
    filelists = os.listdir(args.pretrain_data_dir)
    sortf = []

    # 仅保留符合格式的文件
    valid_files = []
    for f in filelists:
        # 检查文件名是否符合 "病人编号_视频编号_帧编号"
        parts = f.split('_')
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].split('.')[0].isdigit():
            valid_files.append(f)

    if not valid_files:
        raise ValueError("No valid files found in the dataset directory!")

    # 构建排序依据
    for f in valid_files:
        sortf.append(
            int(f.split('_')[0] + '00000') +  # 病人编号
            int(f.split('_')[1] + '000') +   # 视频编号
            int(os.path.splitext(f.split('_')[2])[0])  # 帧编号
        )
    sortf.sort()  # 对排序依据排序

    # 按排序后的顺序构建文件列表
    sort_file = []
    for num in sortf:
        for f in valid_files:
            if num == int(f.split('_')[0] + '00000') + int(f.split('_')[1] + '000') + int(os.path.splitext(f.split('_')[2])[0]):
                sort_file.append(f)

    # 确保排序后列表非空
    if not sort_file:
        raise ValueError("Sorted file list is empty!")

    Max = int(sort_file[-1].split('_')[0])  # 得到最大的病人编号

    num = 1  # 视频编号从 1 开始
    indices = {1: {}}  # 初始化索引，病人编号从 1 开始
    pics = 0  # 当前视频帧计数

    for i, f in enumerate(sort_file):
        mid = []
        ends = str(os.path.splitext(f)[0])  # 提取文件名去掉扩展名
        for j in range(len(ends)):
            if ends[j] == '_':
                mid.append(j)

        # 判断是否属于同一个病人
        if person == int(ends[:mid[0]]):  # 提取病人编号
            if num == int(ends[mid[0] + 1:mid[1]]):  # 属于同一个视频
                pics += 1  # 帧计数增加
            else:  # 切换到下一个视频
                indices[person][num] = pics  # 保存当前视频帧计数
                num += 1  # 视频编号递增
                pics = 1  # 重置帧计数
        else:  # 切换到下一个病人
            indices[person][num] = pics  # 保存当前病人的最后一个视频帧计数
            num = 1  # 视频编号重置为 1
            person = int(ends[:mid[0]])  # 更新病人编号
            pics = 1  # 重置帧计数
            indices[person] = {num: {}}  # 初始化新的病人记录

    # 保存最后一个病人的最后一个视频帧计数
    indices[person][num] = pics

    print("Successfully get indices")
    return indices
