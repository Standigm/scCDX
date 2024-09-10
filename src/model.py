import math

import torch
# import torch.backends.cuda
import torch.nn.functional as F
from torch import nn, optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
try:
    from src.pytorch_tabnet.tab_model import TabNetClassifier
except:
    pass


class MLP(nn.Module):
    def __init__(self, input_dim, fcn_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, fcn_dim[0]),# bias=False),
            self.activation,
            self.dropout,
        )
        
        self.intermediate_layer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fcn_dim[i], fcn_dim[i+1]),# bias=False),
                self.activation,
                self.dropout
            ) for i in range(len(fcn_dim) - 1)
        ])
        
        self.last_layer = nn.Linear(fcn_dim[-1], 1)#, bias=True)
        
    
    def forward(self, x):
        x = self.dropout(x)
        out = self.first_layer(x)
        for layer in self.intermediate_layer:
            out = layer(out)
        out = self.last_layer(out)
        
        return out.squeeze()
    

def setup(args, param_dict, data: Data, mask: torch.Tensor, device: torch.device):
    if args.model == "mlp":
        model = MLP(
            input_dim = data.x.shape[-1],
            fcn_dim = param_dict["fcn_dim"],
            dropout = param_dict["dropout"],
        ).to(device)
            
        optimizer = optim.AdamW(model.parameters(), lr=param_dict["lr"], weight_decay=param_dict["weight_decay"])
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=((len(data.y[mask]) - sum(data.y[mask])) / sum(data.y[mask])).to(device)
        )

        return model, optimizer, criterion
    
    elif args.model == "xgb":
        clf = XGBClassifier(
            scale_pos_weight=((len(data.y[mask]) - sum(data.y[mask])) / sum(data.y[mask])).item(),
            objective='binary:logistic',
            eval_metric='logloss',
            verbosity=0,
            n_jobs=60,
            random_state=args.seed,
            **param_dict,
        )
        
        return clf
    
    elif args.model == "rf":
        clf = RandomForestClassifier(
            n_jobs=60,
            class_weight="balanced",
            n_estimators=param_dict["n_estimators"],
            max_depth=param_dict["max_depth"],
            max_features=param_dict["max_features"],
            min_samples_split=param_dict["min_samples_split"],
            min_samples_leaf=param_dict["min_samples_leaf"],
        )
        
        return clf
    
    elif args.model == "svm":
        clf = SVC(
            probability=True,
            random_state=args.seed,
            class_weight="balanced",
            C=param_dict["C"],
            gamma=param_dict["gamma"],
            kernel=param_dict["kernel"],
        )
        
        return clf
    
    elif args.model == "tabnet":
        clf = TabNetClassifier(
            verbose=1,
            device_name="cuda",
            clip_value=5.,
            optimizer_fn=optim.Adam,
            optimizer_params=dict(lr=param_dict["lr"]),
            scheduler_fn=optim.lr_scheduler.StepLR,
            scheduler_params=dict(step_size=param_dict["step_size"], gamma=0.9),
            n_d=param_dict["n_da"],
            n_a=param_dict["n_da"],
            n_steps=param_dict["n_steps"],
            seed=args.seed,
        )
        
        return clf
