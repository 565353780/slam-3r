pip install ninja cmake

pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

pip install roma gradio matplotlib tqdm opencv-python \
  scipy einops trimesh tensorboard pycuda viser open3d \
  imageio[ffmpeg] scikit-image pyrender

pip install pyglet <2
pip install huggingface-hub[torch] >=0.22

pip install xformers==0.0.28.post2

cd slam3r/pos_embed/curope/
python setup.py build_ext --inplace
