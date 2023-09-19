echo train the phase 1 of CSQNet on StructureNet
/Users/melkor/miniforge3/envs/Melkor/bin/python\
 train.py --name="VNL"\
 --dataset="StructureNet" --perception="csqnet" --training_mode="3d_perception" \
  --phase="1" --batch_size=16 --concept_type="cone" \
 --freeze_perception="True" --lr="0.001" --batch_size="1" --checkpoint_itrs=150\
  --checkpoint_dir="checkpoints/scenelearner/3dpc/KFT_vase_phase0.ckpt"\
