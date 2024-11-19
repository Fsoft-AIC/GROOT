dataset=$1; shift
device=$1; shift
level=${1:-"harder3"}; shift
algo=${1:-"GA"}; shift
iter=${1:-500}; shift

dirpath=$(realpath $(dirname $(dirname $0)))
output_dir="${dirpath}/experiments"
config_file="${dirpath}/scripts/configs/cnn_template.py"
optim_config_path="${dirpath}/scripts/configs/optimize/${algo}.yaml"

if [[ "$dataset" = "AAV" ]]; then
    model_ckpt_path="${dirpath}/experiments/vae_ckpts/AAV.ckpt"
elif [[ "$dataset" = "GFP" ]]; then
    model_ckpt_path="${dirpath}/experiments/vae_ckpts/GFP.ckpt"
fi

pool=1.0
node=2000
neighbor=7

python3 scripts/optimize.py $config_file \
    --model_ckpt_path=$model_ckpt_path \
    --optim_config_path=$optim_config_path \
    --devices=$device --level=$level \
    --dataset=$dataset --output_dir=$output_dir \
    --changes pool_frac=$pool num_iteration=$iter smooth_config.num_node=$node smooth_config.num_neighbor=$neighbor
