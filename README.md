# ClockworkRNN_Porosity_Log_Prediction
A Clockwork RNN model to train on compressional velocity (Vp), formation density (Rhob), gamma ray (Gr), and resistivity (Rt) logs to predict Neutron Porosity (Nphi)

The clockwork RNN model is a tensorflow implementation of Koutnik et al., 2014. 

# This a modified implementation of tomrunia/ClockworkRNN.
The modifications include:
  1. Fixed mask of the hidden layer
  2. Added correct selection of block-rows of the hidden layer for evaluation
  3. Added prediction point-by-point
  4. Added training and validation losses plots
  5. Added learning rate decay plot
