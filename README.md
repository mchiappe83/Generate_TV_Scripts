# Generate_TV_Scripts

Considerations for function get_embed
can also use tf.contrib.layers.embed_sequence(input_data, vocab_size, embed_dim) which maps a sequence of symbols to a sequence of embeddings.

Considerations for build_nn function
I recommend using initializers for weights and bias because error is converging at slow rate. Checkout this code.

logits=tf.contrib.layers.fully_connected(outputs,vocab_size,activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev= 0.1),biases_initializer=tf.zeros_initializer())

Considerations for get_batches function
A faster and more elegant implementation using Numpy would look like such:

    n_batches = int(len(int_text) / (batch_size * seq_length))

    #Drop the last few characters to make only full batches
    xdata = np.array(int_text[: n_batches * batch_size * seq_length])
    ydata = np.array(int_text[1: n_batches * batch_size * seq_length + 1])

    x_batches = np.split(xdata.reshape(batch_size, -1), n_batches, 1)
    y_batches = np.split(ydata.reshape(batch_size, -1), n_batches, 1)

    return np.array(list(zip(x_batches, y_batches)))


Neural Network Training
Checkout the batch size, try to increase it to 128.
For Sequence length , take a look at the Explore the Data section. The average number of words in each line is ~11.5. This value should be about the size of the length of sentences you want to generate while still matching the structure of the data.
Rest all seems perfect

I am providing you the resources for the HyperParameter Optimization. Go through it and will be really helpful for you.
http://neupy.com/2016/12/17/hyperparameter_optimization_for_neural_networks.html
