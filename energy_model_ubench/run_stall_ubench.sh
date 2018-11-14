#BIN=$1

GPUJoule_dir=""

NumCTA="15"
NumThd="32"
NumIter="1000000"


    for bench in $GPUJoule_dir/energy_model_ubench/stall_energy/fadd_l1d_64p/*.out;
    do
        $GPUJoule_dir/nvml/example/power_monitor 5 > $GPUJoule_dir/energy_model_data/stall_energy/"$bench"_"$NumIter"iter_power.txt &
        PM_PID=$!
        $bench $NumCTA $NumThd $NumIter 16 1 &>> $GPUJoule_dir/energy_model_data/stall_energy/"$bench"_"$NumIter"iter_time.txt
        sleep 5
        kill -15 $PM_PID
        sleep 5
    done
