# data setting arguments
batch_size=200
max_length=10

# model architecture arguments
input_size=28
output_size=12
emb_size=56
hidden_size=128
n_layers=2
dropout=0.3

# training arguments
epochs=100
learning_rate=0.001
teacher_forcing_ratio=0.5
clip_threshold=1

python3 main.py \
        --batch_size ${batch_size} \
        --max_length ${max_length} \
        --input_size ${input_size} \
        --output_size ${output_size} \
        --emb_size ${emb_size} \
        --hidden_size ${hidden_size} \
        --n_layers ${n_layers} \
        --dropout ${dropout} \
        --epochs ${epochs} \
        --learning_rate ${learning_rate} \
        --teacher_forcing_ratio ${teacher_forcing_ratio} \
        --clip_threshold ${clip_threshold}
