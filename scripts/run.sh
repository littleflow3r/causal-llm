
python run_bert.py \
--batch_size=16 \
--model_name='Model_1' \
--data_dir='./data/aimed/8' \
--output_dir='./data/aimed/8' \
--cache_dir='./data/aimed/8' \
--plm_root_dir='./resource' \
--plm_name='biobert-large-cased-v1.1' \
--lr=2e-5 \
--epoch=10 \
--dropout=0.1\
--max_len=128\
--gradient_accumulation_steps=2\
--cuda=1