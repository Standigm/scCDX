# How to run the GNN-based state-of-the-art methods

Following commands reproduce GNN-based state-of-the-art methods. 
Notice that result files will be located under the `log/`.
To get the comparison results, you need to change `omics` parameter to `MF+METH+GE+SYS+TOPO` or `MF+METH+GE+SYS+TOPO+scGNN`.

### EMOGI
```bash
cd baselines/GCN
python GCN_cv.py --method EMOGI --omics MF+METH+GE
```

### MTGCN
```bash
cd baselines/GCN
python GCN_cv.py --method MTGCN --omics MF+METH+GE+TOPO
```

### HGDC
```bash
cd baselines/GCN
python GCN_cv.py --method HGDC --omics MF+METH+GE+SYS
```

### MODIG
Due to the out-of-memory issue, automatic mixed precision was used.
Before train the model, you should download the necessary data files from [here](https://doi.org/10.5281/zenodo.7057241) and locate `MODIG/Data`
```bash
cd baselines/MODIG
python main_cv.py --omics MF+METH+GE --scaler # if automatic mixed precision is needed
```

### MRNGCN
```bash
cd baselines/MRNGCN
python train_cv.py --omics MF+METH+GE+TOPO
```
If you want to train using `MF+METH+GE+TOPO+SYS` or `MF+METH+GE+TOPO+SYS+scRaw_all+nonzero_mean`, run the following command first.
```bash
python pretrain.py --omics $OMICS
``` 
