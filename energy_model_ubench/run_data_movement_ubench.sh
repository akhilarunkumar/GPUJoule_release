#BIN=$1

GPUJoule_dir=""

NumCTA="30"
NumThd="1024"
NumIter="1000000"

    for bench in $GPUJoule_dir/energy_model_ubench/data_movement_ept/fadd_l1d_64p/*.out;
    do
        $bench $NumCTA $NumThd $NumIter 32 1 &>> $GPUJoule_dir/energy_model_data/data_movement_energy/l1_cache/"$bench"_"$NumIter"iter_time.txt
        sleep 5
    done

    for bench in $GPUJoule_dir/energy_model_ubench/data_movement_ept/fadd_shared_64p/*.out;
    do
        $bench $NumCTA $NumThd $NumIter 32 1 &>> $GPUJoule_dir/energy_model_data/data_movement_energy/shd_mem/"$bench"_"$NumIter"iter_time.txt
        sleep 5
    done
    
    for bench in $GPUJoule_dir/energy_model_ubench/data_movement_ept/fadd_l2d_64p/*.out;
    do
        $bench $NumCTA $NumThd $NumIter 32 1 &>> $GPUJoule_dir/energy_model_data/data_movement_energy/l2_cache/"$bench"_"$NumIter"iter_time.txt
        sleep 5
    done

    for bench in $GPUJoule_dir/energy_model_ubench/data_movement_ept/fadd_dram_64p/*.out;
    do
        $bench $NumCTA $NumThd $NumIter 32 1 &>> $GPUJoule_dir/energy_model_data/data_movement_energy/dram/"$bench"_"$NumIter"iter_time.txt
        sleep 5
    done

fadd_dram_64p  fadd_l1d_64p  fadd_l2d_64p  fadd_shared_64p
