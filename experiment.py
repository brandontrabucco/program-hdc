import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


PROJECT_BASEDIR = ("G:/My Drive/Academic/Research/" + 
    "Program Synthesis using HDC/")
PLOT_BASEDIR = (PROJECT_BASEDIR + "Results/")
CHECKPOINT_BASEDIR = (PROJECT_BASEDIR + "Backups/")


SEQUENCE_LENGTH = 10
MEMORY_LOCATIONS = 10000
MEMORY_DEPTH = 4
HIDDEN_SIZE = 200
INITIAL_LEARNING_RATE = 0.0001
DECAY_STEPS = SEQUENCE_LENGTH
DECAY_FACTOR = 1.0


PREFIX_CONTROLLER = "controller"
PREFIX_MEMORY = "memory"


EXTENSION_NUMBER = (lambda number: "_" + str(number))
EXTENSION_LOSS = "_loss"
EXTENSION_WEIGHTS = "_weights"
EXTENSION_BIASES = "_biases"
EXTENSION_MEMORY = "_structure"
EXTENSION_LOCATION = "_location"
EXTENSION_CONTENT = "_content"
EXTENSION_QUERY = "_query"
EXTENSION_RESULT = "_result"


COLLECTION_LOSSES = "_losses"
COLLECTION_PARAMETERS = "_params"


def hypercomplex_conjugate(a, n=MEMORY_DEPTH):
    if n == 1:
        a_conjugate = tf.concat([
            tf.strided_slice(
                a, 
                [0, 0], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH])], 1)
        return a_conjugate

    if n == 2:
        a_conjugate = tf.concat([
            tf.strided_slice(
                a, 
                [0, 0], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH]), 
            -1 * tf.strided_slice(
                a, 
                [0, 1], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH])], 1)
        return a_conjugate
    
    if n == 4:
        a_conjugate = tf.concat([
            tf.strided_slice(
                a, 
                [0, 0], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH]), 
            -1 * tf.strided_slice(
                a, 
                [0, 1], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH]), 
            -1 * tf.strided_slice(
                a, 
                [0, 2], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH]), 
            -1 * tf.strided_slice(
                a, 
                [0, 3], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH])], 1)
        return a_conjugate


def hypercomplex_multiply(a, b, n=MEMORY_DEPTH):
    if n == 1:
        b_shifted_0 = tf.concat([
            tf.strided_slice(
                b, 
                [0, 0], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH])], 1)
        return tf.concat([
            tf.reduce_sum((a * b_shifted_0), axis=1, keep_dims=True)], axis=1)
    
    if n == 2:
        b_shifted_0 = tf.concat([
            tf.strided_slice(
                b, 
                [0, 0], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH]), 
            -1 * tf.strided_slice(
                b, 
                [0, 1], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH])], 1)
        b_shifted_1 = tf.concat([
            tf.strided_slice(
                b, 
                [0, 1], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH]), 
            tf.strided_slice(
                b, 
                [0, 0], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH])], 1)
        return tf.concat([
            tf.reduce_sum((a * b_shifted_0), axis=1, keep_dims=True),
            tf.reduce_sum((a * b_shifted_1), axis=1, keep_dims=True)], axis=1)
    
    elif n == 4:
        b_shifted_0 = tf.concat([
            tf.strided_slice(
                b, 
                [0, 0], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH]), 
            -1 * tf.strided_slice(
                b, 
                [0, 1], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH]), 
            -1 * tf.strided_slice(
                b, 
                [0, 2], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH]), 
            -1 * tf.strided_slice(
                b, 
                [0, 3], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH])], 1)
        b_shifted_1 = tf.concat([
            tf.strided_slice(
                b, 
                [0, 1], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH]), 
            tf.strided_slice(
                b, 
                [0, 0], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH]), 
            -1 * tf.strided_slice(
                b, 
                [0, 3], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH]), 
            tf.strided_slice(
                b, 
                [0, 2], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH])], 1)
        b_shifted_2 = tf.concat([
            tf.strided_slice(
                b, 
                [0, 2], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH]), 
            tf.strided_slice(
                b, 
                [0, 3], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH]), 
            tf.strided_slice(
                b, 
                [0, 0], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH]), 
            -1 * tf.strided_slice(
                b, 
                [0, 1], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH])], 1)
        b_shifted_3 = tf.concat([
            tf.strided_slice(
                b, 
                [0, 3], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH]), 
            -1 * tf.strided_slice(
                b, 
                [0, 2], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH]), 
            tf.strided_slice(
                b, 
                [0, 1], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH]), 
            -1 * tf.strided_slice(
                b, 
                [0, 0], 
                [MEMORY_LOCATIONS, MEMORY_DEPTH], 
                strides=[1, MEMORY_DEPTH])], 1)
        return tf.concat([
            tf.reduce_sum((a * b_shifted_0), axis=1, keep_dims=True),
            tf.reduce_sum((a * b_shifted_1), axis=1, keep_dims=True),
            tf.reduce_sum((a * b_shifted_2), axis=1, keep_dims=True),
            tf.reduce_sum((a * b_shifted_3), axis=1, keep_dims=True)], axis=1)


def initialize_weights_cpu(name, shape, standard_deviation=0.01, decay_factor=None, collection=None):

    with tf.device("/cpu:0"):

        weights = tf.get_variable(
            (name + EXTENSION_WEIGHTS),
            shape,
            initializer=tf.truncated_normal_initializer(
                stddev=standard_deviation,
                dtype=tf.float32),
            dtype=tf.float32)

    if decay_factor is not None and collection is not None:

        weight_decay = tf.multiply(
            tf.nn.l2_loss(weights),
            decay_factor,
            name=((name + EXTENSION_WEIGHTS) + EXTENSION_LOSS))
        tf.add_to_collection(collection, weight_decay)

    return weights


def initialize_biases_cpu(name, shape):

    with tf.device("/cpu:0"):

        biases = tf.get_variable(
            (name + EXTENSION_BIASES),
            shape,
            initializer=tf.constant_initializer(1.0),
            dtype=tf.float32)

    return biases


def initialize_memory_cpu(name, shape, decay_factor=None, collection=None):

    with tf.device("/cpu:0"):

        memory = tf.get_variable(
            (name + EXTENSION_MEMORY),
            shape,
            initializer=tf.constant_initializer(0.0),
            dtype=tf.float32)

    if decay_factor is not None and collection is not None:

        memory_decay = tf.multiply(
            tf.nn.l2_loss(memory),
            decay_factor,
            name=((name + EXTENSION_MEMORY) + EXTENSION_LOSS))
        tf.add_to_collection(collection, memory_decay)

    return memory


def reset_kernel():
    global CONTROLLER_INITIALIZED
    global MEMORY_INITIALIZED
    global STEP_INCREMENTED
    CONTROLLER_INITIALIZED = False
    MEMORY_INITIALIZED = False
    STEP_INCREMENTED = False


def controller(data_in):

    global CONTROLLER_INITIALIZED
    activation = data_in


    with tf.variable_scope((PREFIX_CONTROLLER + EXTENSION_NUMBER(1)), reuse=CONTROLLER_INITIALIZED) as scope:

        linear_w = initialize_weights_cpu(
            scope.name, 
            [SEQUENCE_LENGTH + MEMORY_LOCATIONS, HIDDEN_SIZE])


        linear_b = initialize_biases_cpu(
            scope.name, 
            [HIDDEN_SIZE, 1])

        
        activation = tf.nn.relu(tf.add(tf.tensordot(tf.transpose(linear_w), activation, 1), linear_b))


        if not CONTROLLER_INITIALIZED:
            parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope=scope.name)
            for p in parameters:
                tf.add_to_collection((PREFIX_CONTROLLER + COLLECTION_PARAMETERS), p)


    with tf.variable_scope((PREFIX_CONTROLLER + EXTENSION_NUMBER(2)), reuse=CONTROLLER_INITIALIZED) as scope:

        linear_w_location = initialize_weights_cpu(
            (scope.name + EXTENSION_LOCATION), 
            [HIDDEN_SIZE, MEMORY_DEPTH, MEMORY_LOCATIONS])
        linear_b_location = initialize_biases_cpu(
            (scope.name + EXTENSION_LOCATION), 
            [MEMORY_LOCATIONS, MEMORY_DEPTH, 1])
        location = tf.add(tf.tensordot(tf.transpose(linear_w_location), activation, 1), linear_b_location)


        linear_w_content = initialize_weights_cpu(
            (scope.name + EXTENSION_CONTENT), 
            [HIDDEN_SIZE, MEMORY_DEPTH, MEMORY_LOCATIONS])
        linear_b_content = initialize_biases_cpu(
            (scope.name + EXTENSION_CONTENT), 
            [MEMORY_LOCATIONS, MEMORY_DEPTH, 1])
        content = tf.add(tf.tensordot(tf.transpose(linear_w_content), activation, 1), linear_b_content)


        linear_w_query_l = initialize_weights_cpu(
            (scope.name + EXTENSION_QUERY  + "_l"), 
            [HIDDEN_SIZE, MEMORY_DEPTH, MEMORY_LOCATIONS])
        linear_b_query_l = initialize_biases_cpu(
            (scope.name + EXTENSION_QUERY  + "_l"), 
            [MEMORY_LOCATIONS, MEMORY_DEPTH, 1])
        query_l = tf.add(tf.tensordot(tf.transpose(linear_w_query_l), activation, 1), linear_b_query_l)


        linear_w_query_c = initialize_weights_cpu(
            (scope.name + EXTENSION_QUERY  + "_c"), 
            [HIDDEN_SIZE, MEMORY_DEPTH, MEMORY_LOCATIONS])
        linear_b_query_c = initialize_biases_cpu(
            (scope.name + EXTENSION_QUERY  + "_c"), 
            [MEMORY_LOCATIONS, MEMORY_DEPTH, 1])
        query_c = tf.add(tf.tensordot(tf.transpose(linear_w_query_c), activation, 1), linear_b_query_c)


        linear_w_output = initialize_weights_cpu(
            (scope.name + EXTENSION_RESULT), 
            [HIDDEN_SIZE, SEQUENCE_LENGTH])
        linear_b_output = initialize_biases_cpu(
            (scope.name + EXTENSION_RESULT), 
            [SEQUENCE_LENGTH, 1])
        output = tf.add(tf.tensordot(tf.transpose(linear_w_output), activation, 1), linear_b_output)
        

        if not CONTROLLER_INITIALIZED:
            parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope=scope.name)
            for p in parameters:
                tf.add_to_collection((PREFIX_CONTROLLER + COLLECTION_PARAMETERS), p)


    CONTROLLER_INITIALIZED = True
    return location, content, query_l, query_c, output


def update_memory(key_batch, value_batch, query_l_batch, query_c_batch):

    global MEMORY_INITIALIZED


    with tf.variable_scope((PREFIX_MEMORY + EXTENSION_NUMBER(1)), reuse=MEMORY_INITIALIZED) as scope:

        memory = initialize_memory_cpu(
            scope.name, 
            [MEMORY_LOCATIONS, MEMORY_DEPTH, 1])

        
        memory = memory + hypercomplex_multiply(key_batch, value_batch)
        output = tf.reduce_sum(
            hypercomplex_multiply(
                hypercomplex_conjugate(query_l_batch), 
                hypercomplex_multiply(
                    memory, 
                    hypercomplex_conjugate(query_c_batch))), axis=1)

    MEMORY_INITIALIZED = True
    return output


def similarity_loss(prediction, labels, collection):

    huber_loss = tf.losses.huber_loss(labels, prediction)

    
    tf.add_to_collection(collection, huber_loss)

    return huber_loss


def minimize(loss, parameters):
    
    global STEP_INCREMENTED


    global_step = tf.train.get_or_create_global_step()


    learning_rate = tf.train.exponential_decay(
        INITIAL_LEARNING_RATE,
        global_step,
        DECAY_STEPS,
        DECAY_FACTOR,
        staircase=True)


    optimizer = tf.train.AdamOptimizer(learning_rate)


    if not STEP_INCREMENTED:

        gradient = optimizer.minimize(
            loss, 
            var_list=parameters,
            global_step=global_step)


    else:

        gradient = optimizer.minimize(
            loss, 
            var_list=parameters)


    STEP_INCREMENTED = True
    return gradient


def train_model(num_epochs=1):
    
    reset_kernel()


    num_steps = num_epochs * SEQUENCE_LENGTH


    with tf.Graph().as_default():

        output_indices = tf.concat([tf.range(SEQUENCE_LENGTH // 2), tf.range(SEQUENCE_LENGTH // 2)], 0)
        input_indices = tf.constant([0 if i % 2 == 0 else 1 for i in range(SEQUENCE_LENGTH)], dtype=tf.int32)
        output_tensor = tf.one_hot(output_indices, SEQUENCE_LENGTH)
        input_tensor = tf.one_hot(input_indices, SEQUENCE_LENGTH)

        
        feedback = tf.zeros([MEMORY_LOCATIONS, 1])
        predicted_sequence = tf.zeros([SEQUENCE_LENGTH, 0])


        for i in range(SEQUENCE_LENGTH):

            sliced_input = tf.transpose(tf.strided_slice(
                input_tensor,
                [i, 0],
                [SEQUENCE_LENGTH, SEQUENCE_LENGTH],
                strides=[SEQUENCE_LENGTH, 1]))
            sliced_output = tf.transpose(tf.strided_slice(
                output_tensor,
                [i, 0],
                [SEQUENCE_LENGTH, SEQUENCE_LENGTH],
                strides=[SEQUENCE_LENGTH, 1]))


            location, content, query_l, query_c, prediction = controller(
                tf.concat([sliced_input, feedback], 0))
            predicted_sequence = tf.concat([predicted_sequence, prediction], 1)

            prediction_loss = similarity_loss(
                sliced_output, 
                prediction,
                (PREFIX_CONTROLLER + COLLECTION_LOSSES))
            feedback = update_memory(location, content, query_l, query_c)


        prediced_indices = tf.argmax(predicted_sequence, axis=0)
        controller_parameters = tf.get_collection(PREFIX_CONTROLLER + COLLECTION_PARAMETERS)
        controller_loss = tf.add_n(tf.get_collection(PREFIX_CONTROLLER + COLLECTION_LOSSES))
        controller_gradient = minimize(controller_loss, controller_parameters)


        model_saver = tf.train.Saver(var_list=controller_parameters)


        with tf.train.MonitoredTrainingSession(hooks=[
                tf.train.StopAtStepHook(num_steps=num_steps),
                tf.train.CheckpointSaverHook(
                    CHECKPOINT_BASEDIR,
                    save_steps=SEQUENCE_LENGTH*10,
                    saver=model_saver)]) as session:

                while not session.should_stop():

                    _, indices, loss = session.run([controller_gradient, prediced_indices, controller_loss])
                    print(indices, loss)