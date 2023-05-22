import time
from typing import Optional
from dpu_utils.utils import RichPath
import os
from data_utils import LeeVdataset
from train import run_models
from torch.utils.data import DataLoader
from train_utils import get_train_cli_arg_parser

def create_datapath(dir,path):
    data_path = os.path.join(dir,path)
    data_path = RichPath.create(data_path)
    return data_path

def my_collate(batch):
    data = [item for item in batch]
    return data

def make_run_id(model_name: str, task_name: str, run_name: Optional[str] = None) -> str:
    """Choose a run ID, based on the --run-name parameter and the current time."""
    if run_name is not None:
        return run_name
    else:
        return "%s_%s__%s" % (model_name, task_name, time.strftime("%Y-%m-%d_%H-%M-%S"))

def log_line(log_file: str, msg: str):
    with open(log_file, "a") as log_fh:
        log_fh.write(msg + "\n")
    print(msg)



def run_train_from_args(args):

    data_path = create_datapath(args.data_path, "train.jsonl.gz")
    data_path1 = create_datapath(args.data_path, "valid.jsonl.gz")
    data_path2 = create_datapath(args.data_path, "test.jsonl.gz")
    data_train = LeeVdataset(data_path)
    data_valid = LeeVdataset(data_path1)
    data_test = LeeVdataset(data_path2)
    dl_train = DataLoader(data_train,batch_size=128,shuffle=True,collate_fn=my_collate)
    dl_valid = DataLoader(data_valid, batch_size=128, shuffle=True, collate_fn=my_collate)
    dl_test = DataLoader(data_test, batch_size=128, shuffle=True, collate_fn=my_collate)
    run_models(dl_train,dl_valid,dl_test,args)



if __name__ == "__main__":
    parser =get_train_cli_arg_parser()
    args, potential_hyperdrive_args = parser.parse_known_args()

    run_train_from_args(args)