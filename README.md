# OCT-symulation-for-WTC-11-hiPS-cell-s

This code is used to simulate OCT images for any 3D data. I have included several mask sizes that must match the spectral size of the 3D image in question. The file contains masks in EW_int_X_Y.mat format. This means that the mask has the dimension (Y, X, X) for example EW_int_528_66.mat has the size (66, 528, 528).

In the Calculation.py file, I have included the code that determines the OCT for a given image. And in the Support.ipynb file, I included code to help check 3D images using the napari library.