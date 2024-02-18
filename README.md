# 3D-Face-Reconstruction
# To see the Face expression transfer go to the reconstruction folder.
# If you want to run the code, please make sure that you have created a virtual environment and installed the required libraries.
# How to install the required libraries?
- `python3 -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`
# How to run the code?
- Copy the files from `https://drive.google.com/drive/folders/1FhLSHv8to8n-919K5ZsINMMsiekusldH` and create a data directory.
- Please make sure that CmakeList.txt has the correct path to the Libraries.
- In the directory, 
- `mkdir build`
- `cd build`
- `cmake ..`
- `make`
- `./face_reconstruction ../images/micheal.jpeg ../images/wolverine.jpg`
- See the reconstruction folder for the results.
- The left path is the source image and the right path is the target image.
# Team Members:
- Ramandika Pranamulia 
- Rachmadio Noval Lazuardi 

- Utku Saglam
# Some Figures:
<img width="626" alt="micheal_scream_all_rec" src="https://github.com/utkusaglm/3DFace-Reconstruction/assets/58150504/e351611c-e5a1-4a24-a0f0-c8e5ea29e3a9"><img width="540" alt="micheal_all_rec" src="https://github.com/utkusaglm/3DFace-Reconstruction/assets/58150504/f7b2525b-ea53-4726-9943-b182c6c1e533">
