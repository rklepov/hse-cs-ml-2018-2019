Georgia Tech face database (128Mb) contains images of 50 people taken in two 
or three sessions between 06/01/99 and 11/15/99 at the Center for Signal and 
Image Processing at  Georgia Institute of Technology. All people in the database 
are represented by 15 color JPEG images with cluttered background taken at resolution 
640x480 pixels. The average size of the faces in these images is 150x150 pixels. 
The images are stored in 50 directories s1, ..., s50. In each directory there 
are 15 images 01.jpg, ..., 15.jpg corresponding to one person in the database. 

Each image is manually labeled to determine the position of the face in the image.
The label files contain four integers that describe the coordinates of the face rectangles 
and a string (s1, ..., s50) indicating the identity of the face. Assuming that the upper 
left corner of the image has the coordinates (0,0), the numbers correspond to the 
x_left, y_top, x_right, y_bottom coordinates of the face rectangle. 

The label files are named as follows: 
lab001-lab015 correspond to files 01.jpg,...,15.jpg  in s01
lab021-lab035 correspond to files 01.jpg,...,15.jpg  in s02
lab041-lab055 correspond to files 01.jpg,...,15.jpg  in s03
lab061-lab075 correspond to files 01.jpg,...,15.jpg  in s04
....

