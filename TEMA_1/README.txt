1. The libraries required to run the project including the full version of each library

python == 3.9.15
numpy == 1.23.4
opencv-python == 4.6.0.66

2. how to run each task and where to look for the output file

task 1:
script: TEMA_1.py
at the top of the script , at lines 5-17 there is a description on how to
properly replace the three paths to make the script work. There are 3 paths that need to be customized for each machine. 
path_solutii: the path where the solutions files will be created

path_antrenare: the path from where the program will read the images and procces them to create the solutions files.

path_templates: the path from where the program will read the templates for each piece. In the project folder you will also find a folder called templates which contains the best handpicked templates to use for this project.

After all these paths are properly configured, the script can be run and the solutions files will be created in the folder that is specified in path_solutii.
From then on, this folder path can be used in evalueaza_solutii.py to check the solutions. The script takes about 1 minute to run.

