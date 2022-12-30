sudo apt update &&
  sudo apt upgrade &&
  sudo apt install build-essential software-properties-common cmake unzip pkg-config &&
  sudo apt install -y gcc-7 g++-7 &&
  sudo apt install screen &&
  sudo apt install libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev &&
  sudo apt install libjpeg-dev libpng-dev libtiff-dev &&
  sudo apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev &&
  sudo apt install libxvidcore-dev libx264-dev &&
  sudo apt install libopenblas-dev libatlas-base-dev liblapack-dev gfortran &&
  sudo apt install libhdf5-serial-dev &&
  sudo apt install python3-dev python3-tk python-imaging-tk &&
  sudo apt install libgtk-3-dev &&
  pip install -r requirement.txt
