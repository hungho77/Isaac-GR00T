python gr00t/eval/open_loop_eval.py \
    --dataset-path /mnt/data/sftp/data/hunght23/DATA_ZOO/aloha_left_arm_pick_carrot_put_cup_added_recovery \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path tmp/mobile_aloha_finetune_left_arm_pick_carrot_put_cup_21072026/checkpoint-2000 \
    --traj-ids 0 \
    --save_plot_path tmp/open_loop_eval/traj_0 \
    --action-horizon 16  # ensure this is within the delta_indices of action's modality config.