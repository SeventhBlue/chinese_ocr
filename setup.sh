#pip install numpy==1.16
#pip install scipy matplotlib pillow
#pip install easydict opencv-python
#pip install keras==2.2.4
#pip install h5py PyYAML
#pip install cython==0.24

# for gpu
#pip install tensorflow-gpu==1.14.0
chmod +x ./ctpn/lib/utils/make.sh
cd ./ctpn/lib/utils/ && ./make.sh

# for cpu
# pip install tensorflow==1.3.0
# chmod +x ./ctpn/lib/utils/make_cpu.sh
# cd ./ctpn/lib/utils/ && ./make_cpu.sh
