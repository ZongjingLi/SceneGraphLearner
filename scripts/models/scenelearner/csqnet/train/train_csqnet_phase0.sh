echo train the phase 0 of CSQNet on StructureNet
/Users/melkor/miniforge3/envs/Melkor/bin/python\
 train.py --name="VNL"\
 --dataset="StructureNet" --perception="csqnet" --training_mode="3d_perception" \
  --phase="0" --batch_size=16 --concept_type="cone" \
    --checkpoint_dir="checkpoints/scenelearner/3dpc/VNL_3d_perception_structure_csqnet_phase0.pth"\