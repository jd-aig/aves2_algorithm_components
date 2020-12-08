# PlatformNlp
PlatformNlp 框架的架构来源于[Fairseq](https://github.com/pytorch/fairseq)，模型部分借鉴了google-rearch的[Bert](https://github.com/google-research/bert)


PlatformNlp 能够基于完成常见的multi-class文本分类、multi-label文本分类，相似度计算，命名实体识别等任务。整体的使用方法分为三步：1: 转为tf-record文件 2:完成训练 3:完成验证

---
其中文本分类所需的数据集如下所示:（分隔符为,） (multi-label任务将不同的label用+拼接 如 不新鲜+包装差)
<table>
  <tr>
    <th width=50%, bgcolor=#999999 >Text</th> 
    <th width="50%", bgcolor=#999999>Label</th>
  </tr>
</table>

其中文本相似度计算所需的数据集如下所示:（分隔符为,）
<table>
  <tr>
    <th width=33%, bgcolor=#999999 >Text 1</th> 
    <th width=33%, bgcolor=#999999>Text 2</th>
    <th width="34%", bgcolor=#999999>Label</th>
  </tr>
</table>

其中命名实体识别所需的数据集如下所示: （分隔符为空格" "） (支持BIO标记法) (不同句子之间用空行分割)
<table>
  <tr>
    <th width=50%, bgcolor=#999999 >Word</th> 
    <th width=50%, bgcolor=#999999>Label</th>
  </tr>
</table>


## Get Started in 60 Seconds

以训练一个文本分类任务为例，使用PlatformNlp下的预处理脚本来完成转换tf-record的过程：

```sh
python3 preprocess.py --data_file ../all_data/multi_class/data.csv --dict_file ../all_data/dict/vocab.txt --output_dir ../all_data/eval_data/ --type train --word_format char --label_file ../all_data/data_dir/labels.pkl --output_file ../all_data/eval_data/multi_class.tfrecord --max_seq_length 200
```

接着训练模型，通过--task参数指定任务类型，通过--arch参数指定模型，并将所需参数传递进去：
```sh
python3 train.py --train_data_file ../all_data/data_dir/multi_class.tfrecord --output_dir ../all_data/train_output/ --max_seq_length 200 --arch textcnn --embedding_size 128 --filter_sizes 2,3,4 --num_filters 128 --l2_reg_lambda 0.1 --drop_prob 0.1 --initializer_range 0.1 --label_file ../all_data/data_dir/labels.pkl --batch_size 32 --vocab_size 21128 --num_classes 2 --epoch 1 --device_map "0" --criterion multi_class_cross_entropy
```

接着进行模型的验证，通过--metrics参数指定所需的参数验证：

```sh
python3 eval.py --test_data_file ../all_data/data_dir/multi_class.tfrecord --output_dir ../all_data/test_output/ --model_dir ../all_data/train_output/ --init_checkpoint model.ckpt-2497 --max_seq_length 200 --arch textcnn --embedding_size 128 --filter_sizes 2,3,4 --num_filters 128 --l2_reg_lambda 0.1 --drop_prob 0.1 --initializer_range 0.1 --label_file ../all_data/data_dir/labels.pkl --batch_size 32 --vocab_size 21128 --num_classes 2 --epoch 1 --device_map "0" --criterion multi_class_cross_entropy --metrics multi_class_cross_entry_metrics

```


## Models

1. [DSSM](https://github.com/NTMC-Community/MatchZoo/tree/master/matchzoo/models/dssm.py): this model is an implementation of <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf">Learning Deep Structured Semantic Models for Web Search using Clickthrough Data</a>

2. [CDSSM](https://github.com/NTMC-Community/MatchZoo/tree/master/matchzoo/models/cdssm.py): this model is an implementation of <a href="https://www.microsoft.com/en-us/research/publication/learning-semantic-representations-using-convolutional-neural-networks-for-web-search/">Learning Semantic Representations Using Convolutional Neural Networks for Web Search</a>

3. [Word2vec](https://github.com/tensorflow/tensorflow/blob/r0.12/tensorflow/examples/tutorials/word2vec/word2vec_basic.py): this model is an implementation of <a href="https://arxiv.org/abs/1301.3781v3">Efficient Estimation of Word Representations in Vector Space</a>

4. [Bert](https://github.com/google-research/bert): this model is an implementation of <a href="https://arxiv.org/abs/1810.04805">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a>

5. [TextCnn](https://github.com/dennybritz/cnn-text-classification-tf): this model is an implementation of <a href="https://arxiv.org/abs/1408.5882">Convolutional Neural Networks for Sentence Classification</a>

## reference:

1. [Ner] the evaluate code come from https://github.com/spyysalo/conlleval.py

2. [Fairseq] https://github.com/pytorch/fairseq


