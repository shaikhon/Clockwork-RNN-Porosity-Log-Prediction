
class Config(object):

    output_dir = "./output/"

    # Clockwork RNN parameters
    # periods       = [1, 2, 3, 8, 12, 16, 32, 64]
    # periods        = [1, 2, 3, 4, 6, 8, 16, 24, 32]
    # periods        = [1, 2, 3, 4, 6, 8, 16, 24, 32, 64]
    # periods        = [1, 1, 1, 2, 2, 3, 4, 5, 6, 8]
    periods        = [1, 2, 3, 4, 5, 8, 16, 24, 32] 

    num_steps   = 0
    num_input   = 100
    num_hidden  = len(periods) * 42
    num_output  = 0

    # Optmization parameters
    num_epochs          = 1000
    batch_size          = 512
    optimizer     = "adam"
    max_norm_gradient   = 10.0

    # Exponential Learning rate decay schedule
    learning_rate       = 0.500e-4          #1.e-4
    learning_rate_decay = 0.955             #.995
    learning_rate_step  = 1000              #500
    learning_rate_min   = 1.00e-5

    # fraction of test examples of total examples such that: split = # of test examples / # of total examples
    split = .07454
