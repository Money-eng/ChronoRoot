import pandas as pd
from pathlib import Path

root_path = Path('.') 
csv_files = list(root_path.glob('temp/out/*/*/epoch_*/*/*/Results 0/*.csv'))
print(f"Nombre de fichiers CSV trouvés : {len(csv_files)}")

dataframes_growth = []
dataframes_morpho = []

for file_path in csv_files:

    plant_num_folder = file_path.parents[1]
    sub_box_name = file_path.parents[2]
    epoch_folder = file_path.parents[3]
    loss_folder = file_path.parents[4]
    
   
    plant_num = int(plant_num_folder.name.split('_')[1])
    sub_box_name = int(sub_box_name.name)
    epoch_val = int(epoch_folder.name.split('_')[1])
    loss_name = loss_folder.name
    box_name = loss_folder.parent.name
    
    print(f"Processing file: {file_path} | box: {box_name} | loss: {loss_name} | epoch: {epoch_val} | img: {sub_box_name} | plant: {plant_num}")

    df_temp = pd.read_csv(file_path)
    
    df_temp['box_name'] = box_name
    df_temp['loss_name'] = loss_name
    df_temp['epoch'] = epoch_val
    df_temp['img_num'] = sub_box_name
    df_temp['plant_num'] = plant_num
    
    # put col at the beginning
    cols = df_temp.columns.tolist()
    new_order = ['box_name', 'loss_name', 'epoch', 'img_num', 'plant_num'] + [col for col in cols if col not in ['box_name', 'loss_name', 'epoch', 'img_num', 'plant_num']]
    df_temp = df_temp[new_order]

    if file_path.name == 'GrowthSpeeds.csv':
        dataframes_growth.append(df_temp)
        
    elif file_path.name == 'Postprocessed.csv':
        dataframes_morpho.append(df_temp)


if dataframes_growth:
    df_global_growth = pd.concat(dataframes_growth, ignore_index=True)
    output_growth = 'GLOBAL_DYNAMICS.csv'
    df_global_growth.to_csv(output_growth, index=False)

if dataframes_morpho:
    # list all postprocessed dataframes
    df_post_list = [df for df in dataframes_morpho]
    
    # Concatenate all postprocessed dataframes
    df_post_all = pd.concat(df_post_list, ignore_index=True) if df_post_list else pd.DataFrame()

    output_morpho = 'GLOBAL_MORPHOLOGY.csv'
    df_post_all.to_csv(output_morpho, index=False)
    