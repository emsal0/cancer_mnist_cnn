import numpy as np
import tensorflow as tf
import pandas as pd
import sys
import time

from pyhocon import ConfigFactory

CONF = ConfigFactory.parse_file('data.conf')
FILENAME = CONF.get('dataset.filename')
MODEL_DIR = CONF.get('model.dir')

TRAIN_PERCENTAGE = 0.9


def cnn_model_fn(features, labels, mode):
    input_layer = tf.convert_to_tensor(features["x"])

    conv1 = tf.layers.conv2d(
        input_layer, # input
        filters=32, # of filters
        kernel_size=[3,3], # dimension of filter
        padding='same',
        activation=tf.nn.relu
    )
    conv2 = tf.layers.conv2d(
        conv1, # input
        filters=32, # of filters
        kernel_size=[3,3], # dimension of filter
        padding='same',
        activation=tf.nn.relu
    )

    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    drop1 = tf.layers.dropout(inputs=pool1, rate=0.25, training=True)

    conv3 = tf.layers.conv2d(
        drop1,
        filters=64,
        kernel_size=[3,3],
        padding="same",
        activation=tf.nn.relu
    )
    conv4 = tf.layers.conv2d(
        conv3,
        filters=64,
        kernel_size=[3,3],
        padding="same",
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2,2], strides=2)

    pool2_flat = tf.layers.Flatten()(pool2)

    dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=True)

    logits = tf.layers.dense(inputs=dropout, units=7)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss = loss,
            global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op = train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(argv):
    dat = pd.read_csv(FILENAME)
    train_labels = np.asarray(dat['label'], dtype=np.int32)
    print 'train_labels: ', train_labels
    print 'labels: ', dat['label'].unique(), '(type %s)' % type(dat['label'].unique())
    num_labels = len(dat['label'].unique())

    X = dat.drop(['label'], axis=1).values
    print X.shape

    print X[0,:]

    Xn = np.reshape(X, (-1, 28, 28 * 3))
    print 'Xn.shape: ', Xn.shape
    print Xn[0,:,:], Xn[0,:,:].shape

    Xf = np.zeros((Xn.shape[0],28,28,3))

    """
    The following for loop splits the image into its red, green, and blue channels.
    """
    for i in range(3):
        Xf[:, :, :, i] = Xn[:, :, np.mod(np.arange(Xn.shape[2]), 3) == i]

    last_train_index = int(Xf.shape[0] * TRAIN_PERCENTAGE)

    np.random.shuffle(Xf)

    Xf_train = Xf[ :last_train_index, :, :, :]
    y_train = train_labels[:last_train_index]
    Xf_test = Xf[last_train_index:, :, :, :]
    y_test = train_labels[last_train_index:]

    tf.logging.set_verbosity(tf.logging.INFO)

    classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=MODEL_DIR)

    print classifier.config.cluster_spec
    time.sleep(2)

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": Xf_train},
        y=y_train,
        batch_size=100,
        num_epochs=None,
        shuffle=False
    )

    if argv[1] != 'test':
        classifier.train(
            input_fn=train_input_fn,
            steps=20000,
            hooks=[logging_hook]
        )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": Xf_test},
        y=y_test,
        num_epochs=1,
        shuffle=False
    )
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print eval_results

if __name__ == '__main__':
    main(sys.argv)
