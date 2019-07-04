#!/usr/bin/env bash

# Download SQuAD
SQUAD_DIR=SQuAD
mkdir -p $SQUAD_DIR
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $SQUAD_DIR/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $SQUAD_DIR/dev-v1.1.json
mv vocab.msgpack SQUAD_DIR

# Download GloVe
GLOVE_DIR=glove
mkdir -p $GLOVE_DIR
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O $GLOVE_DIR/glove.840B.300d.zip
unzip $GLOVE_DIR/glove.840B.300d.zip -d $GLOVE_DIR

# sent2vec
git clone https://github.com/epfml/sent2vec.git
cd sent2vec/src/
python setup.py build_ext
pip install .

# get wiki_bigram
cd ..
cat << 'EOT' >> ~/.bash_aliases
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}
EOT

source ~/.bash_aliases
gdrive_download 0B6VhzidiLvjSaER5YkJUdWdPWU0 wiki_bigrams.bin
cd ..
