python scripts/run_train.py --model_type rand --dataset SST-1 --optimizer Adam
python scripts/run_train.py --model_type static --dataset SST-1 --optimizer Adam
python scripts/run_train.py --model_type non-static --dataset SST-1 --optimizer Adadelta
python scripts/run_train.py --model_type multichannel --dataset SST-1 --optimizer Adadelta


python scripts/run_train.py --model_type rand --dataset SST-2 --optimizer Adam
python scripts/run_train.py --model_type static --dataset SST-2 --optimizer Adam
python scripts/run_train.py --model_type non-static --dataset SST-2 --optimizer Adadelta
python scripts/run_train.py --model_type multichannel --dataset SST-2 --optimizer Adadelta
