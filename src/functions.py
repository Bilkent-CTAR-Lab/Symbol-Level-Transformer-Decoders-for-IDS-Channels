#import numpy as np
import os
import tensorflow as tf
#from keras import backend as K

def bits_to_int(MSB_num, bits_array):
  # Convert numpy array of bits to binary string
  binary_string = ''.join([str(bit) for bit in bits_array])
  # Convert binary string to integer
  integer = int(binary_string, 2)
  return integer

def save_model_weights(model, path_main, model_name):
  #path_main = '/drive/MyDrive'
  path = os.path.join(path_main, model_name)
  model.save_weights(path, overwrite = True)
  print('Model is succesfully saved to ', path)
  return

def load_model_weights(model, path_main, filepath):
  #path_main = '/drive/MyDrive'
  path = os.path.join(path_main, filepath)
  model = model.load_weights(path)
  print('Model is succesfully loaded from ', path)
  return model

def save_args_to_file(args, directory, filename='parameters.txt'):
  filename = os.path.join(directory, filename)
  with open(filename, 'w') as f:
      for arg, value in vars(args).items():
          f.write(f"{arg}: {value}\n")
  print(f"Arguments saved to {filename}")

@tf.function
def train_step(model, inputs, labels, loss_fn, optimizer, 
               train_acc_metric, train_acc_metric_topk, 
               batch_size=None, symmm=None, ind=None, use_indices=False):
    """
    One training step for the model.

    Args:
        model: Keras model to train
        inputs: training batch
        labels: ground truth labels
        loss_fn: loss function
        optimizer: optimizer
        train_acc_metric: main accuracy metric
        train_acc_metric_topk: top-k accuracy metric
        batch_size: used when reshaping labels (only needed if use_indices=True)
        symmm: symbol length (only needed if use_indices=True)
        ind: indices for selecting subset (optional, only used if use_indices=True)
        use_indices: whether to reshape + gather based on indices
    """
    with tf.GradientTape() as tape:
        logits = model(inputs)

        if use_indices:
            labels = tf.reshape(labels, [batch_size, symmm])
            selected_labels = tf.gather(labels, ind, axis=1)
            selected_logits = tf.gather(logits, ind, axis=1)
            loss = loss_fn(labels, logits)
        else:
            selected_labels = labels
            selected_logits = logits
            loss = loss_fn(labels, logits)

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_acc_metric.update_state(selected_labels, selected_logits)
    train_acc_metric_topk.update_state(selected_labels, selected_logits)

    return loss

  