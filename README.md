

# AVES2 Algorithm Components

AVES2 Algorithm Components为AVES2的开发者提供了一些算法组件的示例，可以帮助开发者快速构建自己的算法组件。

## 运行环境准备
AVES2 Algorithm Components是运行在AVES2上的，因此需要先创建kubernetes集群，准备好TensorFlow或者PyTorch的镜像（具体版本及其他依赖取决于算法对环境的要求），然后启动AVES2服务，详细步骤请参考AVES2的说明。

## 客户端环境准备
在提交任务时，需要在客户端运行，因此需要先准备AVES2Client的环境，并且配置好AVES2的服务地址以及访问存储时所需要的鉴权信息，详细步骤请参考AVESClient的说明。

## 准备数据
运行时需要的数据和代码需要放到集群的存储，例如对象存储中。

## 创建任务描述文件

这里是一个任务描述文件的例子
```text
job_name: 'mobilenet-v2 image classification'
debug: true
distribute_type: Null  # Null, HOROVOD, TF_PS
image: ai-image.jd.com/jupyter/tensorflow:1.12-nb5.7-ubuntu16.04-py3.6
resource_spec:
  worker:
    count: 1
    cpu: 4
    mem: 2  # 2G
    gpu: 1
running_spec:
  source_code:  # 源代码相关配置
    storage_mode: OSSFile
    storage_conf:
      path: s3://aves2/opensource/components/mobilenet-v2-14/
  envs: []                              # 用户可以设置环境变量
  cmd: "python3 model_train_new.py"  # 启动命令
  normal_args:                      # 超参数
    - name: model_name
      value: mobilenet_v2_14
    - name: num_class
      value: 37
    - name: batch_size
      value: 4
    - name: epochs
      value: 1
    # - name: lr
    #   value: 0.01
  input_args:                           # 输入参数
    - name: train_data_dir
      data_conf:
        storage_mode: OSSFile
        storage_conf:
          path: s3://aves2/opensource/datasets/classification/pettfrecord/train/
    - name: validation_data_dir
      data_conf:
        storage_mode: OSSFile
        storage_conf:
          path: s3://aves2/opensource/datasets/classification/pettfrecord/test/
    - name: pre_trained_model_ckpt_path
      data_conf:
        storage_mode: OSSFile
        storage_conf:
          path: s3://aves2/opensource/pretrainedmodels/image_classification/
  output_args:                          # 输出参数
    - name: result_dir
      data_conf:
        storage_mode: OSSFile
        storage_conf:
          path: s3://aves2/opensource/test/mobilenet-v2/out/
    - name: checkpoint_dir
      data_conf:
        storage_mode: OSSFile
        storage_conf:
          path: s3://aves2/opensource/test/mobilenet-v2/checkpoint/

```

## Reference
* <a href="https://github.com/jd-aig/aves2">AVES2</a>:  Aves2是一个分布式机器学习模型训练引擎，支持用户提交训练任务到Kubernetes或docker swarm集群。
* <a href="https://github.com/jd-aig/aves2_client">AVES2Client</a> Aves2Client是Aves2平台的命令行工具，支持用户进行提交、查看、删除任务等基本等任务管理功能。