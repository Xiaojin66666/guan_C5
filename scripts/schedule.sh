#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py experiment=ftd "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"

python src/train.py experiment=ftd "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"

python src/train.py experiment=ftd "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"

python src/train.py experiment=ftd "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"

python src/train.py experiment=ftd "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"

python src/train.py experiment=ftd "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"

python src/train.py experiment=ftd "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"

python src/train.py experiment=ftd data.action='BTB' "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"

python src/train.py experiment=ftd data.action='BTB' "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"

python src/train.py experiment=ftd data.action='BTB' "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"

python src/train.py experiment=ftd data.action='BTB' "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"

python src/train.py experiment=ftd data.action='BTB' "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"

python src/train.py experiment=ftd data.action='BTB' "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"

python src/train.py experiment=ftd data.action='BTB' "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"

python src/train.py experiment=ftd "data.altitude='&H10000F0V211X21'" "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"

python src/train.py experiment=ftd "data.altitude='&H10000F0V211X21'" "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"

python src/train.py experiment=ftd "data.altitude='&H10000F0V211X21'" "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"

python src/train.py experiment=ftd "data.altitude='&H10000F0V211X21'" "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"

python src/train.py experiment=ftd "data.altitude='&H10000F0V211X21'" "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"

python src/train.py experiment=ftd "data.altitude='&H10000F0V211X21'" "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"

python src/train.py experiment=ftd "data.altitude='&H10000F0V211X21'" "model.net.hidden_channels=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 10]"


#python src/train.py experiment=ftd data.action='BTB' "data.altitude='&H10000F0V211X21'" "data.scaler._target_= 'sklearn.preprocessing.MinMaxScaler'"

#python src/train.py experiment=ftd "data.altitude='&H10000F0V211X21'" "data.scaler._target_= 'sklearn.preprocessing.MinMaxScaler'"

#python src/train.py experiment=ftd "data.altitude='&H35000F0M81X21'" "data.scaler._target_= 'sklearn.preprocessing.MinMaxScaler'"

#python src/train.py experiment=ftd data.action='BTB' "data.scaler._target_= 'sklearn.preprocessing.MinMaxScaler'"


#python src/train.py experiment=ftd "data.altitude='&H27000F0M61X21'" "data.scaler._target_= 'sklearn.preprocessing.MinMaxScaler'"
