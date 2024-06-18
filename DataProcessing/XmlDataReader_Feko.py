import os
import xml.etree.ElementTree as ET
import torch

import os
import xml.etree.ElementTree as ET
import torch

def XmlDataReader_Feko(file_path, file_name, missing_num, samples_num, output_path):
    try:
        tree = ET.parse(os.path.join(file_path, file_name))
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"XML parsing error: {e}")
    except FileNotFoundError as e:
        raise ValueError(f"File not found: {e}")
    except Exception as e:
        raise ValueError(f"An error occurred while reading the XML file: {e}")

    variables_list = []
    
    for file_elem in root.findall('file'):  
        variables = {}
        has_required_variables = True      
        for variable_elem in file_elem.findall('variable'):  
            name = variable_elem.get('name')
            try:
                value = float(variable_elem.get('value'))
            except ValueError:
                has_required_variables = False
                break
            variables[name] = value
        
        if has_required_variables and 'sx' in variables and 'sy' in variables and 'sz' in variables:
            variables_list.append([variables['sx'], variables['sy'], variables['sz']])

    if not variables_list:
        raise ValueError("No valid 'sx', 'sy', 'sz' variables found in the XML file.")
    
    print(f'Parameters read successfully!')

    variables_tensor = torch.tensor(variables_list, dtype=torch.float64)
    
    full_set = set(range(1, samples_num+1))
    keep_list = list(full_set - missing_num)
    keep_idx = [i-1 for i in keep_list]
    
    variables_tensor = variables_tensor[keep_idx, :]  
    print(f'Parameters tensor shape: {variables_tensor.shape}')
    
    torch.save(variables_tensor, os.path.join(output_path, f'{file_name}_param.pt'))    
    print(f'Save tensor to {output_path} successfully!')
    
    return variables_tensor