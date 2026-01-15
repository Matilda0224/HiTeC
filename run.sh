for dname in citeseer cora
do
    for model_type in  'tricl_ngs'
    do
        for sub_rate in 0.3
        do
            for s in 3
            do
                                    python train.py \
                                    --dataset $dname \
                                    --device 0 \
                                    --model_type $model_type\
                                    --s_walk $s \
                                    --num_seeds 3\
                                    --sub_rate $sub_rate \
                                    >> result/${dname}_${model_type}_s${s}.out 
            done    
        done
    done
done