import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


PROJECT_BASEDIR = ("G:/My Drive/Academic/Research/" +
    "Program Synthesis using HDC/")
PLOT_BASEDIR = (PROJECT_BASEDIR + "Results/")
CHECKPOINT_BASEDIR = (PROJECT_BASEDIR + "Backups/")


SEQUENCE_LENGTH = 6
MEMORY_LOCATIONS = 32
MEMORY_DEPTH = 8
HIDDEN_SIZE = 256
INITIAL_LEARNING_RATE = 0.0001
UNCERTAINTY_FACTOR = 1.0
DECAY_STEPS = SEQUENCE_LENGTH
DECAY_FACTOR = 1.0


PREFIX_CONTROLLER = "controller"
EXTENSION_LOSS = "_loss"
EXTENSION_WEIGHTS = "_weights"
EXTENSION_BIASES = "_biases"
EXTENSION_PREDICTION = "_prediction"
EXTENSION_UNCERTAINTY = "_uncertainty"


PREFIX_MEMORY = "memory"
EXTENSION_CELL = "_cell"
EXTENSION_LOCATION = "_location"
EXTENSION_CONTENT = "_content"
EXTENSION_ADD = "_add"
EXTENSION_SELECT = "_select"


LAYER_NUMBER = (lambda number: "_" + str(number))
COLLECTION_LOSSES = "_losses"
COLLECTION_PARAMETERS = "_params"


def hypercomplex_conjugate_2d(a, m=MEMORY_DEPTH):

    if m == 1:
        return a

    else:

        return tf.concat([

            hypercomplex_conjugate_2d(
                tf.slice(
                    a,
                    [0, 0],
                    [MEMORY_LOCATIONS, (m - 1)]),
                m=(m - 1)),

            (-1 * tf.slice(
                a,
                [0, (m - 1)],
                [MEMORY_LOCATIONS, 1]))], 1)


def hypercomplex_conjugate_3d(a, m=MEMORY_DEPTH):

    if m == 1:
        return a

    else:

        return tf.concat([

            hypercomplex_conjugate_3d(
                tf.slice(
                    a,
                    [0, 0, 0],
                    [MEMORY_LOCATIONS, MEMORY_LOCATIONS, (m - 1)]),
                m=(m - 1)),

            (-1 * tf.slice(
                a,
                [0, 0, (m - 1)],
                [MEMORY_LOCATIONS, MEMORY_LOCATIONS, 1]))], 2)


def hypercomplex_multiply_2d(a, b, m=MEMORY_DEPTH):

    if m == 1:
        return a * b

    else:

        def cayley_dickson(p, q, r, s):

            return tf.concat([
                (hypercomplex_multiply_2d(
                    p,
                    r,
                    m=(m//2)) -
                hypercomplex_multiply_2d(
                    hypercomplex_conjugate_2d(s, m=(m//2)),
                    q,
                    m=(m//2))),
                (hypercomplex_multiply_2d(
                    s,
                    p,
                    m=(m//2)) +
                hypercomplex_multiply_2d(
                    q,
                    hypercomplex_conjugate_2d(r, m=(m//2)),
                    m=(m//2)))], 1)

        return cayley_dickson(

            tf.slice(
                a,
                [0, 0],
                [MEMORY_LOCATIONS, (m//2)]),

            tf.slice(
                a,
                [0, (m//2)],
                [MEMORY_LOCATIONS, (m//2)]),

            tf.slice(
                b,
                [0, 0],
                [MEMORY_LOCATIONS, (m//2)]),

            tf.slice(
                b,
                [0, (m//2)],
                [MEMORY_LOCATIONS, (m//2)]))


def hypercomplex_multiply_3d(a, b, m=MEMORY_DEPTH):

    if m == 1:
        return a * b

    else:

        def cayley_dickson(p, q, r, s):

            return tf.concat([
                (hypercomplex_multiply_3d(
                    p,
                    r,
                    m=(m//2)) -
                hypercomplex_multiply_3d(
                    hypercomplex_conjugate_3d(s, m=(m//2)),
                    q,
                    m=(m//2))),
                (hypercomplex_multiply_3d(
                    s,
                    p,
                    m=(m//2)) +
                hypercomplex_multiply_3d(
                    q,
                    hypercomplex_conjugate_3d(r, m=(m//2)),
                    m=(m//2)))], 2)

        return cayley_dickson(

            tf.slice(
                a,
                [0, 0, 0],
                [MEMORY_LOCATIONS, MEMORY_LOCATIONS, (m//2)]),

            tf.slice(
                a,
                [0, 0, (m//2)],
                [MEMORY_LOCATIONS, MEMORY_LOCATIONS, (m//2)]),

            tf.slice(
                b,
                [0, 0, 0],
                [MEMORY_LOCATIONS, MEMORY_LOCATIONS, (m//2)]),

            tf.slice(
                b,
                [0, 0, (m//2)],
                [MEMORY_LOCATIONS, MEMORY_LOCATIONS, (m//2)]))


def hypercomplex_dft(a, v, n=MEMORY_LOCATIONS, m=MEMORY_DEPTH):

    def hypercomplex_euler():

        coefficients = tf.reshape(tf.tensordot(
            tf.reshape(tf.range(n, dtype=tf.float32), [n, 1]),
            tf.reshape(tf.range(n, dtype=tf.float32), [1, n]), 1), [n, n, 1])
        amplitude = tf.exp(tf.slice(v, [0], [1]) * 2.0 * np.pi * coefficients / n)
        magnitude = tf.sqrt(tf.reduce_sum(tf.square(tf.slice(v, [1], [m - 1]))))

        return tf.concat([
            (amplitude * tf.cos(2.0 * np.pi * coefficients * magnitude / n)),
            (tf.tile(
                amplitude * tf.sin(2.0 * np.pi * coefficients * magnitude / n) / magnitude,
                [1, 1, (m - 1)])) * tf.reshape(tf.slice(v, [1], [m - 1]), [1, 1, (m - 1)])], 2)

    return tf.reduce_sum(
        hypercomplex_multiply_3d(
            1 / hypercomplex_euler(),
            tf.tile(
                tf.reshape(a, [n, 1, m]),
                [1, n, 1]),
            m=m), axis=0) / n


def hypercomplex_idft(a, v, n=MEMORY_LOCATIONS, m=MEMORY_DEPTH):

    def hypercomplex_euler():

        coefficients = tf.reshape(tf.tensordot(
            tf.reshape(tf.range(n, dtype=tf.float32), [n, 1]),
            tf.reshape(tf.range(n, dtype=tf.float32), [1, n]), 1), [n, n, 1])
        amplitude = tf.exp(tf.slice(v, [0], [1]) * 2.0 * np.pi * coefficients / n)
        magnitude = tf.sqrt(tf.reduce_sum(tf.square(tf.slice(v, [1], [m - 1]))))

        return tf.concat([
            (amplitude * tf.cos(2.0 * np.pi * coefficients * magnitude / n)),
            (tf.tile(
                amplitude * tf.sin(2.0 * np.pi * coefficients * magnitude / n) / magnitude,
                [1, 1, (m - 1)])) * tf.reshape(tf.slice(v, [1], [m - 1]), [1, 1, (m - 1)])], 2)

    return tf.reduce_sum(
        hypercomplex_multiply_3d(
            hypercomplex_euler(),
            tf.tile(
                tf.reshape(a, [n, 1, m]),
                [1, n, 1]),
            m=m), axis=0)


def initialize_weights_cpu(
        name,
        shape,
        standard_deviation=0.01,
        decay_factor=None,
        collection=None):

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
            initializer=tf.constant_initializer(10.0),
            dtype=tf.float32)

    return biases


def initialize_memory_cpu(
        name,
        shape,
        decay_factor=None,
        collection=None):

    with tf.device("/cpu:0"):

        memory = tf.get_variable(
            (name + EXTENSION_CELL),
            shape,
            initializer=tf.constant_initializer(0.0),
            dtype=tf.float32)

    if decay_factor is not None and collection is not None:

        memory_decay = tf.multiply(
            tf.nn.l2_loss(memory),
            decay_factor,
            name=((name + EXTENSION_CELL) + EXTENSION_LOSS))
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

    with tf.variable_scope(
        (PREFIX_CONTROLLER + LAYER_NUMBER(1)),
        reuse=CONTROLLER_INITIALIZED) as scope:

        linear_w = initialize_weights_cpu(
            scope.name,
            [SEQUENCE_LENGTH + (MEMORY_LOCATIONS * MEMORY_DEPTH * 2), HIDDEN_SIZE])

        linear_b = initialize_biases_cpu(
            scope.name,
            [HIDDEN_SIZE, 1])

        activation = tf.nn.sigmoid(
            tf.add(
                tf.tensordot(tf.transpose(linear_w), activation, 1),
                linear_b))

        if not CONTROLLER_INITIALIZED:
            parameters = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=scope.name)

            for p in parameters:
                tf.add_to_collection(
                    (PREFIX_CONTROLLER + COLLECTION_PARAMETERS),
                    p)

    with tf.variable_scope(
        (PREFIX_CONTROLLER + LAYER_NUMBER(2)),
        reuse=CONTROLLER_INITIALIZED) as scope:

        linear_add_location_w = initialize_weights_cpu(
            (scope.name + EXTENSION_ADD + EXTENSION_LOCATION),
            [HIDDEN_SIZE, MEMORY_DEPTH, MEMORY_LOCATIONS])
        linear_add_location_b = initialize_biases_cpu(
            (scope.name + EXTENSION_ADD + EXTENSION_LOCATION),
            [MEMORY_LOCATIONS, MEMORY_DEPTH, 1])
        add_location = tf.add(
            tf.tensordot(tf.transpose(linear_add_location_w), activation, 1),
            linear_add_location_b)

        linear_add_content_w = initialize_weights_cpu(
            (scope.name + EXTENSION_ADD + EXTENSION_CONTENT),
            [HIDDEN_SIZE, MEMORY_DEPTH, MEMORY_LOCATIONS])
        linear_add_content_b = initialize_biases_cpu(
            (scope.name + EXTENSION_ADD + EXTENSION_CONTENT),
            [MEMORY_LOCATIONS, MEMORY_DEPTH, 1])
        add_content = tf.add(
            tf.tensordot(tf.transpose(linear_add_content_w), activation, 1),
            linear_add_content_b)

        linear_select_location_w = initialize_weights_cpu(
            (scope.name + EXTENSION_SELECT + EXTENSION_LOCATION),
            [HIDDEN_SIZE, MEMORY_DEPTH, MEMORY_LOCATIONS])
        linear_select_location_b = initialize_biases_cpu(
            (scope.name + EXTENSION_SELECT + EXTENSION_LOCATION),
            [MEMORY_LOCATIONS, MEMORY_DEPTH, 1])
        select_location = tf.add(
            tf.tensordot(tf.transpose(linear_select_location_w), activation, 1),
            linear_select_location_b)

        linear_select_content_w = initialize_weights_cpu(
            (scope.name + EXTENSION_SELECT + EXTENSION_CONTENT),
            [HIDDEN_SIZE, MEMORY_DEPTH, MEMORY_LOCATIONS])
        linear_select_content_b = initialize_biases_cpu(
            (scope.name + EXTENSION_SELECT + EXTENSION_CONTENT),
            [MEMORY_LOCATIONS, MEMORY_DEPTH, 1])
        select_content = tf.add(
            tf.tensordot(tf.transpose(linear_select_content_w), activation, 1),
            linear_select_content_b)

        linear_prediction_w = initialize_weights_cpu(
            (scope.name + EXTENSION_PREDICTION),
            [HIDDEN_SIZE, SEQUENCE_LENGTH])
        linear_prediction_b = initialize_biases_cpu(
            (scope.name + EXTENSION_PREDICTION),
            [SEQUENCE_LENGTH, 1])
        prediction = tf.add(
            tf.tensordot(tf.transpose(linear_prediction_w), activation, 1),
            linear_prediction_b)

        linear_uncertainty_w = initialize_weights_cpu(
            (scope.name + EXTENSION_UNCERTAINTY),
            [HIDDEN_SIZE, 1])
        linear_uncertainty_b = initialize_biases_cpu(
            (scope.name + EXTENSION_UNCERTAINTY),
            [1, 1])
        uncertainty = tf.add(
            tf.tensordot(tf.transpose(linear_uncertainty_w), activation, 1),
            linear_uncertainty_b)

        if not CONTROLLER_INITIALIZED:
            parameters = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=scope.name)
            for p in parameters:
                tf.add_to_collection(
                    (PREFIX_CONTROLLER + COLLECTION_PARAMETERS),
                    p)

    CONTROLLER_INITIALIZED = True
    return (tf.nn.softmax(
                tf.reshape(add_location, [MEMORY_LOCATIONS, MEMORY_DEPTH])),
            tf.reshape(add_content, [MEMORY_LOCATIONS, MEMORY_DEPTH]),
            tf.nn.softmax(
                tf.reshape(select_location, [MEMORY_LOCATIONS, MEMORY_DEPTH])),
            tf.reshape(select_content, [MEMORY_LOCATIONS, MEMORY_DEPTH]),
            tf.nn.softmax(
                prediction),
            uncertainty)


def update_memory(add_location, add_content, select_location, select_content):

    global MEMORY_INITIALIZED

    with tf.variable_scope(
        (PREFIX_MEMORY + LAYER_NUMBER(1)),
        reuse=MEMORY_INITIALIZED) as scope:

        kernel = tf.constant(
            [0 if i == 0 else 1 for i in range(MEMORY_DEPTH)], dtype=tf.float32)

        memory = initialize_memory_cpu(
            scope.name,
            [MEMORY_LOCATIONS, MEMORY_DEPTH])

        memory = memory + hypercomplex_multiply_2d(
            hypercomplex_idft(add_location, kernel),
            add_content)

        recovered_location = hypercomplex_dft(hypercomplex_multiply_2d(
            hypercomplex_conjugate_2d(memory),
            select_content), kernel)

        recovered_content = hypercomplex_multiply_2d(
            hypercomplex_conjugate_2d(hypercomplex_idft(select_location, kernel)),
            memory)

        selection = tf.concat([
            recovered_location,
            recovered_content], 0)

    MEMORY_INITIALIZED = True
    return tf.reshape(
        selection,
        [MEMORY_LOCATIONS * MEMORY_DEPTH * 2, 1])


def similarity_loss(labels, prediction, uncertainty, collection):

    uncertainty = 0.0 * tf.reduce_sum(uncertainty)

    adjusted_loss = (tf.nn.l2_loss(labels - prediction) /
        tf.exp(uncertainty) +
        (UNCERTAINTY_FACTOR * uncertainty))

    tf.add_to_collection(collection, adjusted_loss)

    return adjusted_loss


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

        label_indices = tf.concat([
            tf.range(SEQUENCE_LENGTH // 2),
            tf.range(SEQUENCE_LENGTH // 2)], 0)
        label_tensor = tf.one_hot(label_indices, SEQUENCE_LENGTH)

        data_indices = tf.constant(
            [0 if i % 2 == 0 else 1 for i in range(SEQUENCE_LENGTH)],
            dtype=tf.int32)
        data_tensor = tf.one_hot(data_indices, SEQUENCE_LENGTH)

        feedback = tf.zeros([(MEMORY_LOCATIONS * MEMORY_DEPTH * 2), 1])
        predicted_sequence = tf.zeros([SEQUENCE_LENGTH, 0])
        uncertainty_sequence = tf.zeros([1, 0])

        for i in range(SEQUENCE_LENGTH):

            sliced_data_tensor = tf.transpose(tf.strided_slice(
                data_tensor,
                [i, 0],
                [SEQUENCE_LENGTH, SEQUENCE_LENGTH],
                strides=[SEQUENCE_LENGTH, 1]))

            add_location, add_content, select_location, select_content, prediction, uncertainty = controller(
                tf.concat([sliced_data_tensor, feedback], 0))
            feedback = update_memory(add_location, add_content, select_location, select_content)

            predicted_sequence = tf.concat([predicted_sequence, prediction], 1)
            uncertainty_sequence = tf.concat([uncertainty_sequence, uncertainty], 1)

        prediction_loss = similarity_loss(
            label_tensor,
            predicted_sequence,
            0 * uncertainty_sequence,
            (PREFIX_CONTROLLER + COLLECTION_LOSSES))
        predicted_indices = tf.argmax(predicted_sequence, axis=0)

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

                gradient, indices, loss = session.run([
                    controller_gradient,
                    predicted_indices,
                    controller_loss])
                print(indices, loss)
