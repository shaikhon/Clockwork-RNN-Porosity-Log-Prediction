import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from config import Config


def predict(config):
    path = "output/checkpoints/"
    path = os.path.abspath(path)
    print(path)

    # Read logs
    phi = np.genfromtxt('Phi.txt')
    rho = np.genfromtxt('Rho.txt')
    vp = np.genfromtxt('Vp.txt')
    gr = np.genfromtxt('Gr.txt')
    rt = np.genfromtxt('RT.txt')

    # split examples
    portion = 1-config.split
    train_split = int(portion * vp.shape[0])
    test_split = int(config.split * vp.shape[0]) + train_split

    vp = vp[train_split: test_split, :]
    rho = rho[train_split:test_split, :]
    gr = gr[train_split: test_split, :]
    rt = rt[train_split: test_split, :]
    phi = phi[train_split: test_split, :]

    b = 512         # batch size
    vp = vp[:b, :]
    rho = rho[:b, :]
    gr = gr[:b, :]
    rt = rt[:b, :]
    phi = phi[:b, :]

    # stack examples to match training shapes
    newdata = np.stack((vp, rho, gr, rt), axis=2)

    num = np.random.choice(np.array(range(b)))             # for random plotting
    d = np.array(list(range(1, vp.shape[1]+1))) * .1524    # depth axis

    checkpoint_file = tf.train.latest_checkpoint(path)
    graph = tf.Graph()

    with graph.as_default():
        session_conf = tf.ConfigProto(log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            inputs = graph.get_operation_by_name("inputs").outputs[0]
            predictions = graph.get_operation_by_name('clockwork_cell/predictions').outputs[0]

            pred = sess.run([predictions], feed_dict={inputs: newdata})
            pred = np.squeeze(pred)

            # Plot random example from batch
            plt.interactive(False)
            plt.title("Predicition")
            plt.plot(phi[num, :], d, Linewidth=2, color='b', label="Actual Log")
            plt.plot(pred[num, :], d, Linewidth=2, color='r', ls='--', label="Predicted Log")
            legend = plt.legend(frameon=True)
            plt.ylabel('Depth (m)')
            plt.gca().invert_yaxis()
            plt.show()

            # save text files
            np.savetxt('Predictions.txt', pred)
            np.savetxt('Targets.txt', phi)
    return


if __name__ == "__main__":
    predict(Config())
