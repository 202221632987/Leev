import time
from typing import Optional

from model import LeeV
import torch
import torch.nn as nn
import os
def compute_metrics(out_put,targets):
    TN = 0
    TP = 0
    FP = 0
    FN = 0
    for i in range(out_put.shape[0]):
        predict = out_put[i]
        target = targets[i]
        predicted = predict.argmax()
        targeted = target
        if predicted == 0 and targeted == 0:
            TN += 1
        if predicted == 1 and targeted == 1:
            TP += 1
        if predicted == 1 and targeted == 0:
            FP += 1
        if predicted == 0 and targeted == 1:
            FN += 1

    return TN,TP,FP,FN


def log_metrics(tn,tp,fp,fn):
    accuracy = float(tn+tp)/(float(tn+tp+fn+fp)+float("1e-8"))
    precision = float(tp)/(float(tp+fp)+float("1e-8"))
    recall = float(tp)/(float(tp+fn)+float("1e-8"))
    f1 = 2*(precision*recall)/(precision+recall+float("1e-8"))
    return {"accuracy":accuracy,"precision": precision, "recall" : recall,"f1": f1}


def run_epoch_valid(dataset,model):
    model.eval()
    tn = 0
    tp = 0
    fp = 0
    fn = 0
    for step ,g in enumerate(dataset):
        print("\rstep:",step,end = " ")
        out_put = model(g)
        target = get_target(g)
        TN, TP, FP, FN = compute_metrics(out_put, target)
        tn += TN
        tp += TP
        fp += FP
        fn += FN
    return log_metrics(tn,tp,fp,fn)


def get_target(graphs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target_total = []
    for graph in graphs:
        target = 1 if graph['target_value']==1 else 0
        target_total.append(target)
    return torch.tensor(target_total).to(device)

def make_run_id(dataset:str,model_name: str, task_name: str, run_name: Optional[str] = None) -> str:
    """Choose a run ID, based on the --run-name parameter and the current time."""
    if run_name is not None:
        return run_name
    else:
        return "%s_%s_%s__%s" % (dataset,model_name, task_name, time.strftime("%Y-%m-%d_%H-%M-%S"))

def log_line(log_file: str, msg: str):
    with open(log_file, "a") as log_fh:
        log_fh.write(msg + "\n")
    print(msg)


def make_str(dict):
    string = ""
    for k,v in dict.items():
        string+= f"{k}: {v}, "
    return string

def run_epoch_train(dataset,model,optimizer,criterion):
    model.train()
    total = 0
    tn = 0
    tp = 0
    fp = 0
    fn = 0
    epoch_loss = 0
    for step, g in enumerate(dataset):
        print("\rstep: ",step,end = " ")
        total+=len(g)
        out_put = model(g)
        target = get_target(g)
        loss = criterion(out_put,target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.data
        TN, TP, FP, FN = compute_metrics(out_put, target)
        tn+= TN
        tp+=TP
        fp+=FP
        fn+=FN
    return epoch_loss.item(),log_metrics(tn,tp,fp,fn)


def run_models(dl_train,dl_valid,dl_test,args):
    os.makedirs(args.save_dir,exist_ok=True)
    run_id = make_run_id(args.data_path.split("/")[-1],args.model, args.task)
    log_file = os.path.join(args.save_dir, f"{run_id}.log")
    def log(msg):
        log_line(log_file, msg)
    log("begin to train model")
    patience = 25
    max_epoch = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LeeV(in_dims=200,num_etypes=5)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    initial_valid_results = run_epoch_valid(dl_valid,model)
    log(f"initial_results: {make_str(initial_valid_results)}")
    best_valid_acc = initial_valid_results["f1"]
    torch.save({'model': model.state_dict()}, 'best_model.pth')
    best_valid_epoch = 0
    for epoch in range(1, max_epoch):
        log(f"== Epoch {epoch}")
        loss,result_train = run_epoch_train(dl_train,model,optimizer,criterion)
        log(f"train:{make_str(result_train)}")
        log(f"loss:{loss} ")
        result_valid = run_epoch_valid(dl_valid,model)
        log(f"valid: {make_str(result_valid)}")
        valid_acc = result_valid["f1"]
        if valid_acc > best_valid_acc:
            log(
                f"  (Best epoch so far, target metric decreased to {valid_acc:.5f} from {best_valid_acc:.5f}.)",
            )
            torch.save({'model': model.state_dict()}, 'best_model.pth')
            best_valid_acc = valid_acc
            best_valid_epoch = epoch
        elif epoch - best_valid_epoch >= patience:
            log(
                f"Stopping training after {patience} epochs without "
                f"improvement on validation metric.",
            )
            log (f"Best validation metric: {best_valid_acc}", )
            break
    model_test = LeeV(in_dims=200,num_etypes=5)
    model_test.load_state_dict(torch.load("best_model.pth")["model"],strict=False)
    model_test.to(device)
    test_result = run_epoch_valid(dl_test,model_test)
    log(f"test_result: {test_result}")





