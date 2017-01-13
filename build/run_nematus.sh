NEMATUS_PATH=../../../grad_drop_experiment/wmt16-scripts/sample/data
./nematus --model=./99-drop --source-corpus=$NEMATUS_PATH/corpus.bpe.ro --source-vocab=$NEMATUS_PATH/corpus.bpe.ro.json --target-corpus=$NEMATUS_PATH/corpus.bpe.en --target-vocab=$NEMATUS_PATH/corpus.bpe.en.json
