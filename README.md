# COMP9517


Z5311691 William Wan
z5218542 Luke Cantor
Z5310056 Joel Abraham
Z5364212 Jorawar Singh
Z5359619 Nicholas Petrykowycz


Needed pip installs. 


pip install opencv-python-headless
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install torch
pip install pillow
pip install scikit-image
pip install ultralytics
pip install tkinter

We are sorry if we have missed any pip installs, please install the remaining based on the errors.
Steps to run the classification method on ONE IMAGE. The classification will take 1-2 minutes. Please wait :)

1. Navigate to the CombinedMethods folder
2. Run the combinedMethods.py file
3. Click the "Choose validation annotation file". THIS STEP IS ESSENTIAL. THE PROGRAM WILL NOT RUN WITHOUT A VALID ANNOTATIONS FILE.
4. A file select popup will appear. Navigate to the CombinedMethods folder and select the "valid_annotations" file
5. Click the "Choose Image" button.
6. A file select popup will appear. Navigate to CombinedMethods->valid-> and choose an image to classify. Click the image and press open.
	After this, The program will run. The program is running when you can see the words "Loading" next to the Classification, Score, IoU and Pred to Val center distance.

	Metric explained:
	Classification - Either penguin or turtle
	Score -  The classification score from all the classification methods
	IoU - Intersection over Union. The percent of area the bounding box overlaps with the true bounding box
	Pred to Val center distance - The distance between the pred (predicted) and the val (valid) bounding box centers
	
	Green box - The programs generated bounding box 
	Red box - The valid true bounding box supplied through the valid_annotations file
7. You have successfully ran our classification method on one image!
8. If you want to classify another image, start from step 3. You do NOT have to reselect the validation annotation fil.


Steps to run the classification method on a DIRECTORY.
Running the classification method on a directory will result in a calculation of the Confusion Matrix and the metrics (Accuracy, Precision, Recall, F1 Score, IoU mean, IoU standard deviation, Distance mean, Distance standard deviation

Running the classifier on a folder will NOT display each images results in that directory.

To classify multiple images please use the single image method.

