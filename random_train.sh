mkdir ./learned_model/random_test22 ./npy_files/random_test22 ./result/random_test22
python add_random_data.py -pkl ./feature/feature_v5_fc7.pkl -np ./npy_files/random_test22/random22_1.npy -n 50
python train.py -gpu 0 -np random_test22/random22_1.npy -m random_test22/random22_1
python curve_analysis.py -i ./result/random_test22/random22_1/log -o ./analysis_log/random22.log -n 150
python add_random_data.py -pkl ./feature/feature_v5_fc7.pkl -np ./npy_files/random_test22/random22_2.npy -n 50 -re ./npy_files/random_test22/random22_1.npy
python train.py -gpu 0 -np random_test22/random22_2.npy -m random_test22/random22_2
python curve_analysis.py -i ./result/random_test22/random22_2/log -o ./analysis_log/random22.log -n 300
python add_random_data.py -pkl ./feature/feature_v5_fc7.pkl -np ./npy_files/random_test22/random22_3.npy -n 50 -re ./npy_files/random_test22/random22_2.npy
python train.py -gpu 0 -np random_test22/random22_3.npy -m random_test22/random22_3
python curve_analysis.py -i ./result/random_test22/random22_3/log -o ./analysis_log/random22.log -n 450
python add_random_data.py -pkl ./feature/feature_v5_fc7.pkl -np ./npy_files/random_test22/random22_4.npy -n 50 -re ./npy_files/random_test22/random22_3.npy
python train.py -gpu 0 -np random_test22/random22_4.npy -m random_test22/random22_4
python curve_analysis.py -i ./result/random_test22/random22_4/log -o ./analysis_log/random22.log -n 600
python add_random_data.py -pkl ./feature/feature_v5_fc7.pkl -np ./npy_files/random_test22/random22_5.npy -n 50 -re ./npy_files/random_test22/random22_4.npy
python train.py -gpu 0 -np random_test22/random22_5.npy -m random_test22/random22_5
python curve_analysis.py -i ./result/random_test22/random22_5/log -o ./analysis_log/random22.log -n 750
python add_random_data.py -pkl ./feature/feature_v5_fc7.pkl -np ./npy_files/random_test22/random22_6.npy -n 50 -re ./npy_files/random_test22/random22_5.npy
python train.py -gpu 0 -np random_test22/random22_6.npy -m random_test22/random22_6
python curve_analysis.py -i ./result/random_test22/random22_6/log -o ./analysis_log/random22.log -n 900
