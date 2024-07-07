python run.py --multirun data.sigma=5e-1 encoder.hidden_dim=64,128,256,512,1024,2048,4096 training.favi_mb_size=16 training.lr=1e-4
python run_linear.py --multirun data.sigma=5e-1 encoder.hidden_dim=64,128,256,512,1024,2048,4096 training.favi_mb_size=16 training.lr=1e-4
python plot.py