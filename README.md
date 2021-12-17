# Digit Recogniser
After completing my Intro to Machine Learning (COMP70050) module at Imperial College London I wanted to try building a small digit recogniser using the MNIST dataset. </br>
It uses tensorflow for the model and pygame for the GUI. To run the program run:
```
python3 recogniser.py
```
You can then use these events:
```
c: clears the board of all you've drawn
r: picks a random digit from the MNIST's dataset
SPACE: makes a prediction based on what you've drawn
ESC: quits the GUI

LEFT MOUSEBUTTON: draws
RIGHT MOUSEBUTTON: clears
```

The GUI is initially loaded with a drawing of 5 to showcase the tool. </br>
A problem I have encountered is the MNIST's inability to generalise to digits not seen in the dataset. This could be due to the shading of the pixels being slightly different and the shapes looking different. It seems as though copying how test data looks works perfectly but generalising to any way of drawing a digit can cause it to predict incorrectly. This can also be seen drawing the exact same shape in a different portion of the screen leading to different predictions. However, 2, 3 and 7 seem to be the most accuracte to predict. </br>
If you draw the digit in slightly different ways it usually ends up getting the right prediction - use the `r` command to see MNIST data. A better network may help as mine is a simple feedforward rather than a CNN.

## Model
To run the model there are two options, running with the `-train` flag and running with no flag. Using the `-train` flag will perform a hyperparameter search on the MNIST dataset then run tests and print out the confusion matrix. No flag will load a pre-trained model and perform predictions on the test data, printing out the confusion matrix.
