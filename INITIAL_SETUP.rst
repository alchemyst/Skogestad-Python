==============================================================================
Guide on how to use the functions contained in the Skogestad-Python repository
==============================================================================

When attemtping to run the reproductions you will need to set up a path directory
that will allow Python to use the libraries contained in the Skogestad-Python folder.

For example, if you run the code for reproductions/Figure/Figure_02_02.py you might get
the follwing error: ImportError: cannot import name `tf`
This happens because the utils and utilsplot libraries that contain the functions
needed to execute this code are located in the main Skogestad-Python folder but
Python does not "know" that the functions are stored there.

An explanation can be found at:
https://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows-7
but here is a more specific way to to fix it for this library in Windows:

Go to:
My Computer > Properties > Advanced System Settings > Environment Variables >

Then under "System variables" find the variable "Path"
Select it and click on "Edit..."
Click on "New" and type in the directory where the Skogestad-Python repository folder
is located on your computer. I.e. on my computer the repository is located under:
"C:\ProgramData\Anaconda3\Skogestad-Python"
Click "OK"
The Python example files should then run without error.
