The data files contain training examples for each log type (i.e. Vp, Rhob, Gr, Rt, and Phi). Each file contain a total of 7124 examples stacked vertically, including 6593 training and 531 validation examples. Onces each file is read in python, the shape will be 64x7124.

## Processing applied to the data include:
  1. Removal of rows containing nan values from all logs
  2. Log-transformation to Gr, Rt, and Phi, to make their distributions look more normal
  3. Min-max normalization to the range [-0.9, 0.9]   
  (To learn more about normalization: http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html) 

## Augmentation techniques include:
  1. Doubling the number of examples by flipping the logs
  2. A window of length 64 points and stride of 32 points to extract examples

## Units:
  * Vp   = Km/s
  * Rhob = g/cc
  * Gr   = API
  * Rt   = Ohm.m
  * Phi  = V/V
