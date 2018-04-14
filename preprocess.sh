mkdir data

cd data

wget https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt
wget https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt
wget https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt

cd ..

python process_data.py