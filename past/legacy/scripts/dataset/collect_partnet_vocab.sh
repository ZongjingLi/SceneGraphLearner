Vocab="chair"
echo start to Collect Category:$Vocab Vocab For PartNet Dataset
/Users/melkor/miniforge3/envs/Melkor/bin/python datasets/p3d_dataset/structure_net/collect_vocab.py\
 --mode="geo" --category=$Vocab