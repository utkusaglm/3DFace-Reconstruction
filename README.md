# 3D-Face-Reconstruction
# How to run the code?
# For landmark detection?
- We already provided the landmarks of images. If you want to see the landmarks and visualize it, you can run the following command:
-  In the directory, please create a virtual env:
- `python3 -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`
- `python3 show_landmark.py --name images/<image_name>.<ext>`
# For the 3D face reconstruction?
- Please make sure that CmakeList.txt has the correct path to the Libraries.
- In the directory, `mkdir build`
- `cd build`
- `cmake ..`
- `make`
- .
# Team Members:
- Ramandika Pranamulia (03736495)
- Rachmadio Noval Lazuardi (03766457)
- Utku Saglam (03772729)