python scripts/run_train.py --model_type rand --dataset MR --optimizer Adam
python scripts/run_train.py --model_type rand --dataset SST-1 --optimizer Adam
python scripts/run_train.py --model_type rand --dataset SST-2 --optimizer Adam
python scripts/run_train.py --model_type rand --dataset Subj --optimizer Adam
python scripts/run_train.py --model_type rand --dataset TREC --optimizer Adam
python scripts/run_train.py --model_type rand --dataset CR --optimizer Adam
python scripts/run_train.py --model_type rand --dataset MPQA --optimizer Adam

python scripts/run_train.py --model_type static --dataset MR --optimizer Adam
python scripts/run_train.py --model_type static --dataset SST-1 --optimizer Adam
python scripts/run_train.py --model_type static --dataset SST-2 --optimizer Adam
python scripts/run_train.py --model_type static --dataset Subj --optimizer Adam
python scripts/run_train.py --model_type static --dataset TREC --optimizer Adam
python scripts/run_train.py --model_type static --dataset CR --optimizer Adam
python scripts/run_train.py --model_type static --dataset MPQA --optimizer Adam

python scripts/run_train.py --model_type non-static --dataset MR --optimizer Adadelta
python scripts/run_train.py --model_type non-static --dataset SST-1 --optimizer Adadelta
python scripts/run_train.py --model_type non-static --dataset SST-2 --optimizer Adadelta
python scripts/run_train.py --model_type non-static --dataset Subj --optimizer Adadelta
python scripts/run_train.py --model_type non-static --dataset TREC --optimizer Adadelta
python scripts/run_train.py --model_type non-static --dataset CR --optimizer Adadelta
python scripts/run_train.py --model_type non-static --dataset MPQA --optimizer Adadelta

python scripts/run_train.py --model_type multichannel --dataset MR --optimizer Adadelta
python scripts/run_train.py --model_type multichannel --dataset SST-1 --optimizer Adadelta
python scripts/run_train.py --model_type multichannel --dataset SST-2 --optimizer Adadelta
python scripts/run_train.py --model_type multichannel --dataset Subj --optimizer Adadelta
python scripts/run_train.py --model_type multichannel --dataset TREC --optimizer Adadelta
python scripts/run_train.py --model_type multichannel --dataset CR --optimizer Adadelta
python scripts/run_train.py --model_type multichannel --dataset MPQA --optimizer Adadelta
