## 分布式部署TensorflowMnist手写字体识别任务

### How to run
`git clone git@github.com:leo-mao/dist-mnist.git`
```
git clone git@github.com:leo-mao/dist-mnist.git
python distributed_server-basic.py --job_name ps --task_index 0 --ps_hosts 127.0.0.1:9910 --worker_hosts 127.0.0.1:9900,127.0.0.1:9901
python distributed_server-basic.py --job_name worker --task_index 0 --ps_hosts 127.0.0.1:9910 --worker_hosts 127.0.0.1:9900,127.0.0.1:9901
python distributed_server-basic.py --job_name worker --task_index 1 --ps_hosts 127.0.0.1:9910 --worker_hosts 127.0.0.1:9900,127.0.0.1:9901 
```