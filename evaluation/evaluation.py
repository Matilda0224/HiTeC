import torch
import torch.nn.functional as F
from torch import Tensor

from .logreg import LogReg
import torch.nn as nn
from torch_scatter import scatter_add, scatter
from sklearn.metrics import roc_auc_score as auroc
import copy


def masked_accuracy(logits: Tensor, labels: Tensor):
    if len(logits) == 0:
        return 0
    pred = torch.argmax(logits, dim=1)
    acc = pred.eq(labels).sum() / len(logits) * 100
    return acc.item()


def accuracy(logits: Tensor, labels: Tensor, masks: list[Tensor]):
    accs = []
    for mask in masks:
        acc = masked_accuracy(logits[mask], labels[mask])
        accs.append(acc)
    return accs


def linear_evaluation(z, labels, masks, lr=0.01, max_epoch=100):
    z = z.detach()
    hid_dim, num_classes = z.shape[1], int(labels.max()) + 1

    classifier = LogReg(hid_dim, num_classes).to(z.device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.0)

    for epoch in range(1, max_epoch + 1):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)

        logits = classifier(z[masks[0]])
        loss = F.cross_entropy(logits, labels[masks[0]])
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        classifier.eval()
        logits = classifier(z)
        accs = accuracy(logits, labels, masks)

    return accs
class MLP_HENN(nn.Module) :
    
    def __init__(self, in_dim, hidden_dim, p = 0.5) : 
        super(MLP_HENN, self).__init__() 
        
        self.classifier1 = torch.nn.Linear(in_dim, hidden_dim)
        self.classifier2 = torch.nn.Linear(hidden_dim, 1)
        self.dropouts = torch.nn.Dropout(p = p)
        
    def forward(self, x, target_nodes, target_ids: list) : 
        
        Z = scatter(src = x[target_nodes, :], index = target_ids, dim = 0, reduce = 'sum')
        Z = (self.classifier1(Z)) # No need of Logits
        Z = torch.relu(Z)
        Z = self.dropouts(Z)
        Z = (self.classifier2(Z)) # No need of Logits
        
        return torch.sigmoid(Z).squeeze(-1) # Edge Prediction Probability

def HE_evaluator(model, X, E, n_node, n_edge, test_Vs, test_IDXs, labels, device) : 
    
    with torch.no_grad() : 
        model.eval()
        pred = model(X, test_Vs, test_IDXs).to('cpu').detach().squeeze(-1).numpy()
        score = auroc(labels, pred)
        return score

def train_HE_predictor(X, edge_buckets, 
                       lr=1e-3, epochs=200, device="cuda:0", seed=0):
    
    mlp_model = MLP_HENN(in_dim=X.shape[1], hidden_dim=128).to(device)

    # Clone and move to device
    train_vidx = edge_buckets[0][0].clone().detach().to(device)
    train_eidx = edge_buckets[0][1].clone().detach().to(device)
    train_label = edge_buckets[0][2].clone().detach().to(device)

    valid_vidx = edge_buckets[1][0].clone().detach().to(device)
    valid_eidx = edge_buckets[1][1].clone().detach().to(device)
    valid_label = edge_buckets[1][2].clone().detach().cpu().numpy()

    test_vidx = edge_buckets[2][0].clone().detach().to(device)
    test_eidx = edge_buckets[2][1].clone().detach().to(device)
    test_label = edge_buckets[2][2].clone().detach().cpu().numpy()

    edges = edge_buckets[3].to(device)
    X = X.to(device)

    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=lr, weight_decay=1e-6)
    criterion = torch.nn.BCELoss()

    valid_score = 0.0
    param = copy.deepcopy(mlp_model.state_dict())  # 提前初始化模型参数

    for ep in range(epochs):
        mlp_model.train()
        optimizer.zero_grad()
        pred = mlp_model(X, train_vidx, train_eidx)
        loss = criterion(pred, train_label)
        loss.backward()
        optimizer.step()

        if (ep + 1) % 10 == 0:
            cur_valid = HE_evaluator(model=mlp_model, X=X, E=None,
                                     n_node=None, n_edge=None,
                                     test_Vs=valid_vidx, test_IDXs=valid_eidx,
                                     labels=valid_label, device=device)

            if cur_valid > valid_score:
                valid_score = cur_valid
                param = copy.deepcopy(mlp_model.state_dict())

    # 使用验证集最优模型参数
    mlp_model.load_state_dict(param)

    # 测试集评估
    test_score = HE_evaluator(model=mlp_model, X=X, E=None,
                              n_node=None, n_edge=None,
                              test_Vs=test_vidx, test_IDXs=test_eidx,
                              labels=test_label, device=device)

    # 训练集评估（注意：此时为评估模式）
    train_score = HE_evaluator(model=mlp_model, X=X, E=None,
                               n_node=None, n_edge=None,
                               test_Vs=train_vidx, test_IDXs=train_eidx,
                               labels=train_label.cpu().numpy(), device=device)

    return float(train_score), float(valid_score), float(test_score)

def node_classification_eval(encoder, data, num_splits=20):
    encoder.eval()
    n, _ = encoder(data.prompts_emb, data.hyperedge_index)

    accs = []
    for i in range(num_splits):
        masks = data.generate_random_split(seed=i)
        accs.append(linear_evaluation(n, data.labels, masks))
    return accs 

def edge_prediction_eval(encoder, data, dataset,  num_splits=20, device='cuda:0'):
    encoder.eval()
    with torch.no_grad():
        n, _ = encoder(data.prompts_emb, data.hyperedge_index)
        n = n.detach() 

    e_path = f'tahg_datasets/{dataset}/edge_bucket_cns.pt'
    edge_buckets = torch.load(e_path,  weights_only = False)
    print(f'load edge_buckets from {e_path}')
    accs = []
    for i in range(num_splits):
        c_buckets = edge_buckets[i]
        train_acc, valid_acc, test_acc = train_HE_predictor( X = n, edge_buckets = c_buckets, 
                                                     lr = 0.005, epochs = 300, device = device, seed = i)
        accs.append([train_acc * 100, valid_acc * 100, test_acc *100])
    return accs