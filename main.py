import os
from DataProcessing.OsDataReader_Feko import *
from DataProcessing.XmlDataReader_Feko import *

param_file_path = 'D:/code/AIforCEM/datasets/CurrentParm'
current_file_path = 'D:/code/AIforCEM/datasets/CurrentData'
samples_num = 1000
model_name = 'TeachRoomModel'
read_mode = 'parallel'
param_file_name = 'TeachRoomModel.xml'

param_output_path = 'D:/code/AIforCEM/datasets/TensorFile'
current_output_path = 'D:/code/AIforCEM/datasets/TensorFile'

missing_num = OsDataReader_Feko(current_file_path, samples_num, model_name, read_mode, current_output_path)
variables_tensor = XmlDataReader_Feko(param_file_path, param_file_name, missing_num, samples_num, param_output_path)