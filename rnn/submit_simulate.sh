#!/bin/bash

for init_num_items in 0 1 2 3 4 5 6
do
    sbatch run_simulate.sh ${init_num_items}
done