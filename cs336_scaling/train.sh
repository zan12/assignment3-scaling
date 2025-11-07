for d_model in 64 128 256 512 1024; do
    for num_layers in 2 4 6 8 12 16 24; do
        # for num_heads in 2 4 8 16; do
            # for lr in 1e-4 3e-4 6e-4 1e-3; do
                for compute_budget in 1e13 3e13 6e13 1e14 3e14 6e14 1e15 3e15 6e15 1e16 3e16 6e16  1e17 3e17 6e17 1e18; do
                    echo "$d_model $num_layers $compute_budget"
                    python -m cs336_scaling.train --d_model=$d_model --num_layers=$num_layers --num_heads=2 --compute_budget=$compute_budget --lr=1e-4
                done
            # done
        # done
    done
done
