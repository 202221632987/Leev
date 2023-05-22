from tqdm import tqdm
JOERNPATH="/opt/joern/joern-cli"
dataset = "Reveal"  # Fan Reveal FFmp
root_dir = 'data/raw_code_'+dataset
import subprocess

def parse_source_code_to_dot(file_path, out_dir_cpg='/cpg'):
    root_path = 'data/parse_code_'+dataset
    try :
        os.makedirs(root_path+out_dir_cpg)
    except:
        pass
    out_dir_cpg = root_path + '/cpg/'
    # parse source code into cpg
    print('parseing source code into cpg...')
    shell_str = "sh " + JOERNPATH + "/joern-parse " + file_path
    subprocess.call(shell_str, shell=True)
    print('exporting cpg from cpg root...')
    shell_export_cpg = "sh " + JOERNPATH + "/joern-export " + "--repr cpg14 --out " + out_dir_cpg + file_path.split('/')[1] + os.sep
    subprocess.call(shell_export_cpg, shell=True)

import os
def main_func(source_dir = root_dir, out_dir_cpg="data/parse_code_"+ dataset):
    # all source code files, each file include a .cpp file
    dirs = os.listdir(source_dir)
    for c_folder in tqdm(dirs):
        file_path = source_dir + '/' + c_folder
        cpg_path = out_dir_cpg + '/' +"cpg"+"/"+ c_folder
        if os.path.exists(cpg_path) and len(os.listdir(cpg_path)) > 0:
            print(f'{file_path} file has been processed')
            continue
        print(f'\nstarting to process {file_path}')
        parse_source_code_to_dot(file_path)

if __name__ == "__main__":
    main_func()


