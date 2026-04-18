import scipy.io
import numpy as np
import sys


def mat_to_dat(mat_file, dat_file=None):
    """将 MAT 文件转换为 DAT 文件"""
    try:
        # 读取 MAT 文件
        mat_data = scipy.io.loadmat(mat_file)

        # 如果没有指定输出文件名，使用相同基本名
        if dat_file is None:
            dat_file = mat_file.replace('.mat', '.dat')

        # 查找第一个数组变量
        array_data = None
        for key, value in mat_data.items():
            if isinstance(value, np.ndarray):
                array_data = value
                break

        if array_data is None:
            print(f"错误：在 {mat_file} 中未找到数组数据")
            return False

        # 处理结构化数组
        if array_data.dtype.names is not None:
            print("警告：检测到结构化数组，将尝试转换为字符串格式")
            # 创建一个字符串数组来存储转换后的数据
            str_data = np.empty(array_data.shape, dtype='object')
            for i in range(array_data.size):
                # 将每个记录转换为字符串
                str_data.flat[i] = ' '.join([str(array_data.flat[i][name]) for name in array_data.dtype.names])
            # 保存为文本文件
            np.savetxt(dat_file, str_data, fmt='%s')
        else:
            # 普通数组直接保存
            np.savetxt(dat_file, array_data)

        print(f"成功转换：{mat_file} → {dat_file}")
        return True

    except Exception as e:
        print(f"转换失败：{str(e)}")
        return False


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python mat_to_dat.py <input.mat> [output.dat]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    mat_to_dat(input_file, output_file)
