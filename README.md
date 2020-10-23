# Music controler with gesture recognition

Music controler that uses gesture recognition and prediction for the commands.
1.	Peace – play\
2.	Palm – pause\
3.	Fist – stop\
4.	L – volume down\
5.	Thumbs-up – volume up


### Prerequisited

Read the requirements file for the packages needed to run the this program.

## Usage

In order to control the music first we need to extract the gesture. I did that using the well known method of of background substraction.\
So after you run the program press 'b' without the hand in the region of interest (the green box) in order to do the background substraction.\
After that place your hand in the region of interest and press Space in order to extract the gesture and predict it.\
If you want to do reset the background press 'r'.
