import os
import re
import numpy as np
import torch 
import concurrent.futures
import pandas as pd



def read_file_os(filename, start_key, skip_lines):
    
    data_df = pd.DataFrame()
    with open(filename, 'r', encoding='utf-8') as file:
        data = file.read()
    lines = data.split('\n')
    start_key_index = next((i for i, line in enumerate(lines) if start_key in line), None)
    end_index = next((i for i, line in enumerate(lines[start_key_index:]) if not line), None) + start_key_index
    start_index = start_key_index + skip_lines + 1
    data_lines = lines[start_index:end_index]
    data_df = pd.DataFrame([line.strip().split() for line in data_lines if line.strip()], index=None, dtype=float)

    return data_df
    """
    Read a file and return a pandas DataFrame.
    
    Args:
        filename (str): The path of the file to be read.
        start_key (str): The keyword to search for the start of the data section.
        skip_lines (int): The number of lines to skip after finding the start_key.
        
    Returns:
        pd.DataFrame: A DataFrame containing the data from the specified section of the file.
    """


def check_samples_num(file_path, model_name, file_suffix):
    
    file_list = [file_name for file_name in os.listdir(file_path) if file_name.endswith(file_suffix)]
    num = []
    pattern = re.compile(rf'{re.escape(model_name)}_(\d+)_Currents1\.os') 
    for file_name in file_list:
        match = pattern.match(file_name)
        if match:
            num.append(int(match.group(1)))
            
    return file_list, num
    """
    Check the number of samples in files with a specific suffix in a directory.
    
    Args:
        file_path (str): The path of the directory to search.
        model_name (str): The name of the model to filter files.
        file_suffix (str): The suffix of the files to search for.
        
    Returns:
        Tuple[List[str], List[int]]: A tuple containing a list of file names and a list of corresponding sample numbers.
    """


def DataTransform(df_list, start, end, dtype=np.float64):
    
    df_split = [df.iloc[:, start:end].values for df in df_list]
    df_array = np.stack(df_split, axis=0)
    df_tensor = torch.from_numpy(df_array.astype(dtype))
    
    return df_tensor
    """
    Transform a list of DataFrames into a PyTorch tensor.
    
    Args:
        df_list (List[pd.DataFrame]): A list of DataFrames to be transformed.
        start (int): The starting column index for data extraction.
        end (int): The ending column index for data extraction.
        dtype (np.dtype, optional): The desired data type of the resulting tensor. Defaults to np.float64.
        
    Returns:
        torch.Tensor: A PyTorch tensor containing the extracted data from the DataFrames.
    """
    
    
def OsDataReader_Feko(files_path, samples_num, model_name, read_mode):
    
    e_current_start_key = 'No. of Electric Current Triangle Samples:'
    m_current_start_key = 'No. of Magnetic Current Triangle Samples:'
    skip_lines = 2
    file_name_list, samples_num_list = check_samples_num(files_path, model_name, 'os')
    full_set = set(range(1, samples_num+1))
    num_set = set(samples_num_list)
    missing_num = full_set - num_set
    if missing_num:
        print(f'Missing samples: {missing_num}')
    if read_mode == 'parallel':
        with concurrent.futures.ThreadPoolExecutor() as executor:
            e_current_data_list = list(executor.map(lambda file_name: read_file_os(os.path.join(files_path, file_name), e_current_start_key, skip_lines), file_name_list))
            m_current_data_list = list(executor.map(lambda file_name: read_file_os(os.path.join(files_path, file_name), m_current_start_key, skip_lines), file_name_list))
            print(f'Current read file mode: Parallel')       
    elif read_mode == 'serial':
        e_current_data_list = [read_file_os(os.path.join(files_path, file_name), e_current_start_key, skip_lines) for file_name in file_name_list]
        m_current_data_list = [read_file_os(os.path.join(files_path, file_name), m_current_start_key, skip_lines) for file_name in file_name_list]
        print(f'Current read file mode: Serial')
    else:
        raise ValueError(f"Invalid read_mode: {read_mode}. Expected 'parallel' or 'serial'.")
    print(f'Number of eletric current samples: {len(e_current_data_list)}')
    print(f'Number of magnetic current samples: {len(m_current_data_list)}')
    
    e_location_feature_tensor = DataTransform(e_current_data_list, start=1, end=4, dtype=np.float64)
    m_location_feature_tensor = DataTransform(m_current_data_list, start=1, end=4, dtype=np.float64)
    
    e_current_label_tensor = DataTransform(e_current_data_list, start=4, end=10, dtype=np.float64)
    m_current_label_tensor = DataTransform(m_current_data_list, start=4, end=10, dtype=np.float64)
    
    print(f'Electric current location feature tensor shape: {e_location_feature_tensor.shape}')
    print(f'Magnetic current location feature tensor shape: {m_location_feature_tensor.shape}')
    print(f'Electric current label tensor shape: {e_current_label_tensor.shape}')
    print(f'Magnetic current label tensor shape: {m_current_label_tensor.shape}')
    print(f'Transform data to tensor successfully!')
    
    torch.save(e_location_feature_tensor, os.path.join(files_path, f'{model_name}_e_location_feature.pt'))
    torch.save(m_location_feature_tensor, os.path.join(files_path, f'{model_name}_m_location_feature.pt'))
    
    torch.save(e_current_label_tensor, os.path.join(files_path, f'{model_name}_e_current_label.pt'))
    torch.save(m_current_label_tensor, os.path.join(files_path, f'{model_name}_m_current_label.pt'))
    
    print(f'Save tensor to {files_path} successfully!')
    
    return None
    
    """
    Read OS files from a directory, transform the data into tensors, and save them.
    
    Args:
        files_path (str): The path of the directory containing the OS files.
        samples_num (int): The expected number of samples.
        model_name (str): The name of the model.
        read_mode (str): The mode for reading files, either 'parallel' or 'serial'.
        
    Returns:
        None
    """
    