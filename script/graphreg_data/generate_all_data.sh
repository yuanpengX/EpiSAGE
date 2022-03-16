for i in `seq 1 19`;do
    (python 2_generate_all_data_v2.py chr${i} ;)&
done
