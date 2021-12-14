# Digit Recogniser
After completing my Intro to Machine Learning module at Imperial College London I wanted to try building a small digit recogniser using the MNIST dataset. </br>
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
A problem I have encountered is the MNIST's inability to generalise to digits not seen in the dataset. This is due to the shading of the pixels being slightly different and the shapes looking different. I have tried to make it as similar to MNIST but certain numbers seem to be difficult to test. 2, 3 and 7 seem to be the most accuracte to predict. </br>
If you draw the digit in slightly different ways it usually ends up getting it - use the r command to see MNIST data. A better network may help as mine is a simple feedforward rather than a CNN.