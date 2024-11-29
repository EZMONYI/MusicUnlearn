#!/bin/bash

data_dir=${1:-"lmd_data"}
mono_data_dir=$data_dir/mono/
para_data_dir=$data_dir/para/
save_dir=$data_dir/processed/

python src/generate_lmd_data.py -i "data/LMD-full-MIDI/" -o $data_dir

mkdir -p $data_dir/mono

cp data/dict.*.txt $data_dir/mono/
cp data/dict.*.txt $data_dir/para/

cat $data_dir/para/train.lyric | sed '/^$/d' > $data_dir/mono/train.lyric
cat $data_dir/para/train.melody | sed '/^$/d' > $data_dir/mono/train.melody
cat $data_dir/para/valid.lyric | sed '/^$/d' > $data_dir/mono/valid.lyric
cat $data_dir/para/valid.melody | sed '/^$/d' > $data_dir/mono/valid.melody
cat $data_dir/para/unlearn.lyric | sed '/^$/d' > $data_dir/mono/unlearn.lyric
cat $data_dir/para/unlearn.melody | sed '/^$/d' > $data_dir/mono/unlearn.melody

mkdir -p $save_dir

for lg in lyric melody
do
	python src/preprocess.py \
		--srcdict $mono_data_dir/dict.$lg.txt \
		--trainpref $mono_data_dir/train --validpref $mono_data_dir/valid \
		--unlearnpref $mono_data_dir/unlearn \
		--destdir $save_dir \
		--workers 20 \
		--source-lang $lg \
	# Since we only have a source language, the output file has a None for the
	# target language. Remove this
	for stage in train valid unlearn
	do
		mv $save_dir/$stage.$lg-None.$lg.bin $save_dir/$stage.$lg.bin
		mv $save_dir/$stage.$lg-None.$lg.idx $save_dir/$stage.$lg.idx
	done
done

# Generate Bilingual Data
python src/preprocess.py \
	--source-lang lyric --target-lang melody \
	--trainpref $para_data_dir/train --validpref $para_data_dir/valid \
	--unlearnpref $mono_data_dir/unlearn \
	--destdir $save_dir \
	--srcdict $para_data_dir/dict.lyric.txt \
	--tgtdict $para_data_dir/dict.melody.txt \
	--workers 20

python src/preprocess.py \
	--source-lang melody --target-lang lyric \
	--trainpref $para_data_dir/train --validpref $para_data_dir/valid \
	--destdir $save_dir \
	--srcdict $para_data_dir/dict.melody.txt \
	--tgtdict $para_data_dir/dict.lyric.txt \
	--workers 20
