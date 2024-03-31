import os
import importlib
import sys
from pre_processing import Preprocessing

# change the address if features of different folders are wanted
data_dir = './DataSet/5'

sys.path.append(data_dir)

for filename in os.listdir(data_dir):
    if filename.endswith('.py') and filename != "__init__.py":
        # 不带扩展名的文件名
        module_name = filename[:-3]

        # 动态导入模块
        module = importlib.import_module(module_name)

        data = module.FRAMES
        processor = Preprocessing(data)

        # print(f"{module_name}: ", end="")
        processor.run(False)
        print(",", end="")
