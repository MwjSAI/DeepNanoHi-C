# DeepNanoHi-C

`DeepNanoHi-C` is a novel computational framework designed specifically to analyze scNanoHi-C data, addressing a critical gap in the current analytical toolset.

## Python Dependencies

```
cooler
h5py
numpy
pandas
torch
pytorch_tabnet
scikit_learn
scipy
```

## Data prepration
1. **data.txt**: A tab-separated file containing the following columns:
   - `cell_id`: Identifier for each cell.
   - `chrom1`: Chromosome name for fragment 1.
   - `pos1`: Location of fragment 1.
   - `chrom2`: Chromosome name for fragment 2.
   - `pos2`: Location of fragment 2.
   - `count`: Count value or normalized weight for the interaction between the fragments.

2. **label_info.pickle**: A file that includes the 'cell type' attribute, which stores the category corresponding to each cell ID.

3. **config.json**: The configuration file for the dataset.

## Tutorial
We provide users with comprehensive usage instructions in the `data` folder.
