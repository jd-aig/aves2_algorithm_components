ARGS=`getopt -o : -l "help,output_path:,tfrecords_dir:,pretrainmodel_dir:,learning_rate:,epochs:,batch_size:" -- "$@"`
eval set -- "${ARGS}"

declare -A run_map=()

HELP_STR="Usage: This script use with args..."

while true;
do
    case "$1" in
        --output_path)
            run_map["output"]="$2"
            shift 2
            ;;
        --tfrecords_dir)
            run_map["tfrecords"]="$2"
            shift 2
            ;;
        --pretrainmodel_dir)
            run_map["pretrainmodel"]="$2"
            shift 2
            ;;
        --learning_rate)
            run_map["learning_rate"]="$2"
            shift 2
            ;;
        --epochs)
            run_map["epochs"]="$2"
            shift 2
            ;;
        --batch_size)
            run_map["batch_size"]="$2"
            shift 2
            ;;
        --help)
            echo -e ${HELP_STR}
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error"
            exit 1
            ;;
    esac
done
export PYTHONPATH=$PYTHONPATH:/opt:/opt/slim:/opt/tensorflow-yolov3-master


CONFIG_FILE="ssd_mobilenet_v2_coco.config"
cp -f ${CONFIG_FILE}.template ${CONFIG_FILE}

label_map="LABEL_MAP"
train_tfrecords="TRAIN_TFRECORDS"
val_tfrecords="VAL_TFRECORDS"
pretrain_model="PRETRAIN_MODEL"
feature_extractor="feature_extractor"
num_classes="@NUM_CLASSES@"
learing_rate="@learning_rate@"
decay_steps="@decay_steps@"
steps_1="@steps_1@"
steps_2="@steps_2@"
decay_factor="@decay_factor@"
num_examples="@eval_num_examples@"
batch_size="@batch_size@"

echo ${run_map['tfrecords']}
sed -i "s:#${label_map}#:${run_map['tfrecords']}ImageSets/label_map.pbtxt:g" ${CONFIG_FILE}
sed -i "s:#${train_tfrecords}#:${run_map['tfrecords']}train.tfrecords:g" ${CONFIG_FILE}
sed -i "s:#${val_tfrecords}#:${run_map['tfrecords']}val.tfrecords:g" ${CONFIG_FILE}
sed -i "s:#${pretrain_model}#:${run_map['pretrainmodel']}model.ckpt:g" ${CONFIG_FILE}
sed -i "s:#${feature_extractor}#:ssd_mobilenet_v2:g" ${CONFIG_FILE}

num_classes_val=20
decay_steps_val=2000
decay_factor_val=0.9
num_examples_val=5000
num_train_steps_val=50000

batch_size_val=${run_map['batch_size']}

python cal_params.py --data_dir ${run_map['tfrecords']} --epochs ${run_map["epochs"]} --batch_size ${batch_size_val}
i=0
for line in $(cat cal_params.txt)
do
    #echo $i
    #echo $line
    case $i in
        0)
        num_classes_val=$line
        echo "i == 0"
        echo ${num_classes_val}
        ;;
        1)
        decay_steps_val=$line
        echo "i == 1"
        echo ${decay_steps_val}
        ;;
        2)
        decay_factor_val=$line
        echo "i == 2"
        echo ${decay_factor_val}
        ;;
        3)
        num_examples_val=$line
        echo "i == 3"
        echo ${num_examples_val}
        ;;
        4)
        num_train_steps_val=$line
        echo "i == 4"
        echo ${num_train_steps_val}
        ;;
    esac
    let i+=1
done

echo ${num_classes_val}
echo ${decay_steps_val}
echo ${decay_factor_val}
echo ${num_examples_val}
echo ${num_train_steps_val}
echo ${batch_size_val}

sed -i "s:${num_classes}:${num_classes_val}:g" ${CONFIG_FILE}
sed -i "s:${learing_rate}:${run_map['learning_rate']}:g" ${CONFIG_FILE}
sed -i "s:${decay_steps}:${decay_steps_val}:g" ${CONFIG_FILE}
sed -i "s:${decay_factor}:${decay_factor_val}:g" ${CONFIG_FILE}
sed -i "s:${num_examples}:${num_examples_val}:g" ${CONFIG_FILE}
sed -i "s:${batch_size}:${batch_size_val}:g" ${CONFIG_FILE}

python model_main.py --pipeline_config_path=${CONFIG_FILE} --model_dir=${run_map['output']} --num_train_steps=${num_train_steps_val} --sample_1_of_n_eval_examples=1 --alsologtostderr
rm -f ${CONFIG_FILE}
rm -f cal_params.txt
mkdir ${run_map['output']}/export_pb/
mv ${run_map['output']}/model.ckpt-*.data-00000-of-00001 ${run_map['output']}/model.ckpt.data-00000-of-00001
mv ${run_map['output']}/model.ckpt-*.index ${run_map['output']}/model.ckpt.index
mv ${run_map['output']}/model.ckpt-*.meta ${run_map['output']}/model.ckpt.meta
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=${run_map['output']}/pipeline.config
#TRAINED_CKPT_PREFIX=${run_map['output']}/model.ckpt
TRAINED_CKPT_PREFIX=${run_map['output']}model.ckpt
EXPORT_DIR=${run_map['output']}/export_pb/
python export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
#rm -rf ${run_map['output']}/export_pb/saved_model
#mv ${run_map['output']}/export/Servo/* ${run_map['output']}/export_pb/saved_model
rm -rf ${run_map['output']}/export
rm -rf ${run_map['output']}/events.*
rm -rf ${run_map['output']}/model*
rm -rf ${run_map['output']}/pipeline.config
rm -rf ${run_map['output']}/graph.pbtxt
rm -rf ${run_map['output']}/eval_eval
rm -rf ${run_map['output']}/checkpoint
mv ${run_map['output']}/export_pb/* ${run_map['output']}/
rm -r ${run_map['output']}/export_pb
