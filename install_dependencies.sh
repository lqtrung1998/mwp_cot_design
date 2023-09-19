pip3 install datasets==2.13.2
pip3 install deepspeed==0.10.3
pip3 install transformers==4.30.2
pip3 install accelerate==0.20.3
pip3 install sigfig==1.3.3
pip3 install wolframclient
sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6  -y

# hdfs dfs -get hdfs://haruna/home/byte_ailab_litg/user/xiaoran.jin/ssat_train/wolfram/* ./
sudo bash WolframEngine_13.2.0_LINUX.sh  < /dev/null