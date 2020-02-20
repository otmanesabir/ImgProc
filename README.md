# ImgProc-HW1

Image Processing (ImgProc) 
Image Processing Course Spring 2020
Homework 1 "MORPHOLOGICAL OPENING AND CLOSING"  

## NOTICE 
I'm not sure if the opening and closing are correct. The professor says reverse the structuring element
but the internet doesn't. I made a little dilation reverse which returns
the same values s the normal one, which makes sense? But I'll try adding an erosion
which shouldn't return something similar.  For now, I used the normal ones and exported the results to enjoy the cool
pics!! 


## TO DO

- ~~Create function to parse command line arguments~~
- ~~Create function to parse the input files and return matrices~~
- ~~Generate the Structuring Element text files~~
- ~~Implement ersosion~~
- ~~Implement dilation~~~
- Implement opening
- Implement closing 


## IMPORTANT

### Output format

FILE FORMAT 'comma separated values (CSV)', for example (5x3 definition domain):
  
  1, 2, 3, 4, 5\n  
  5, 3, 0, 0, 0\n  
  1, 3, 5, 7, 5\n  
  \end-of-file

### EXPERIMENTS: 

11+4 pairs of input image f and SE to apply (output file names specified after '&#8594;') for erosions and dilations:

> erosion of f1 by square of size 3 &#8594; ef1_e1.txt  

> erosion of f2 by horizontal line (|) of size 3 &#8594; ef2_e2.txt 
  
> erosion of f3 by horizontal line (-) of size 3 &#8594; ef3_e2.txt

> erosion of f3 by square of size 5 &#8594; ef3_e3.txt

> erosion of f3 by backward diagonal (\) of size 9 &#8594; ef3_e4.txt
 
> erosion of f3 by forward diagonal (/) of size 9 &#8594; ef3_e5.txt

> dilation of f3 by square of size 5 &#8594; df3_d3.txt

> dilation of f3 by backward diagonal of size 9 &#8594; df3_d4.txt
  
> dilation of f3 by forward diagonal of size 9 &#8594; df3_d5.txt

> opening though dilation of 'ef3_e3.txt' by square of size 5 &#8594; of3_o3.txt

> opening though dilation of 'ef3_e4.txt' by backward diagonal of size 9 &#8594; of3_o4.txt

> one self-chosen experiment with an asymmetric SE (cf. above, add input/output file names to the respective report section)

 
> at least three self-chosen images (cf. above, add input/output file names to the respective report section)  

### TASKS:

* Write a (command line) program that performs either an erosion or a dilation of a two-dimensional gray-scale image 'f' with respect to a (symmetric, odd-sized) structuring element 'SE'. The program should accept four command line arguments: 
 - one switch to choose between erosion and dilation (alternatively, you can implement two programs, one for each operation):
    - d = dilation 
    - e = erosion
 - a first file name argument (SE).  The structuring element SE shall follow the CSV (comma separated values, text file) format, cf. below, and must have odd pixel size (e.g. 3x3, 3x1, 5x5, etc.)
 - a second file name argument (f).  The input file f shall follow the CSV format, cf. below.
 - a third file name argument (f_out).  The output file shall follow the CSV format, cf. below.

* Generate (e.g. manually with a text editor) the required SE text files (SE1.txt, SE2.txt, etc.) for the 11+4 "EXPERIMENTS" listed below. Provide the chosen SE files with your hw submission.

* On output, provide a single CSV file for the eroded/dilated image in the same format and same definition and value domains as the input image 'f'. The output file names for the 12 experiments are defined below.
 
* Hints: 
 - As discussed in the course, the center of the (odd sized) SE definition domain corresponds by convention to vertex (0,0).
 - Assume t_max (maximum gray value) outside of the definition domain of f in case of erosion for simple border handling
 - Assume 0 (minimum gray value) outside of the definition domain of f in case of dilation for simple border handling

* For the large image 'f3.txt', also a png-version is available for your convenience. Try to display the processing results of f3 also as a gray value graphics, in order to assess the plausibility of the result. Submit a screenshot or additional png (or other std graphics file format) file with the result.

* All SEs defined below are symmetric and contain the origin (0,0). Choose one asymmetric SE not containing the origin and demonstrate that the chosen border handling then can introduce artifacts. Describe your actions and findings in the report.

* Finally, do meaningful openings and closings on at least three photographs or 2D images of your choice (min. size: 100x100 pixels). 'Meaningful': Shall be motivated from a real-world perspective. The three chosen problems shall be very different in nature. Write in the report (cf. below) a description of each of the three problems, purpose of the operations to be applied, choice of structuring elements, and provide the output as graphics file.


Hints: 
 - the first experiment would be called from the command line as follows: "YourProgram e SE1.txt f1.txt ef1_e1.txt"
 - the combination of erosion followed by dilation for the same SE is called 'opening'. See what it does by comparing original and result, and try different combinations and sizes for your own interest. 


PROGRAMMING LANGUAGES ALLOWED:
C, C++, Python
