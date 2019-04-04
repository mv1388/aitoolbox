from keras import backend as K
# from keras.engine.topology import Layer, InputSpec
from keras import layers
from keras.layers import recurrent, Lambda, Multiply, Add
from keras.models import Model
from keras.regularizers import L1L2

from sklearn.datasets import make_classification
import random

from AIToolbox.cloud.AWS.model_save import KerasS3ModelSaver
from AIToolbox.cloud.AWS.results_save import S3ResultsSaver
from AIToolbox.experiment_save.result_package.basic_packages import ClassificationResultPackage
from AIToolbox.experiment_save.training_history import TrainingHistory


class RepeatVector4D(layers.Layer):
    def __init__(self, n, **kwargs):
        self.n = n
        # self.input_spec = [InputSpec(ndim=3)]
        super(RepeatVector4D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n, input_shape[1], input_shape[2])

    def call(self, x, mask=None):
        x = K.expand_dims(x, 1)
        pattern = K.stack([1, self.n, 1, 1])
        return K.tile(x, pattern)


class RepeatVector4D_2(layers.Layer):
    def __init__(self, n, **kwargs):
        self.n = n
        # self.input_spec = [InputSpec(ndim=3)]
        super(RepeatVector4D_2, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.n, input_shape[2])

    def call(self, x, mask=None):
        x = K.expand_dims(x, 2)
        pattern = K.stack([1, 1, self.n, 1])
        return K.tile(x, pattern)


def build_FastQA_RNN_concat_model_GLOVE(RNN, story_maxlen, query_maxlen, vocab_size, span_max_idx, span_max_len,
                                        EMBED_HIDDEN_SIZE, dropout_rate):
    """
    RNN deep learning model using only knowledge text and question as the input

    Args:
        RNN:
        story_maxlen:
        query_maxlen:
        vocab_size:
        EMBED_HIDDEN_SIZE:
        dropout_rate:

    Returns:

    """
    print('Build model...')

    """ Question proc """
    question = layers.Input(shape=(query_maxlen,), dtype='int32')
    emb_question = layers.Embedding(vocab_size, 50, trainable=False)(question)
    encoded_question = layers.Dropout(dropout_rate)(emb_question)
    qg_e = layers.Dense(50, activation='sigmoid')(encoded_question)
    qg_e_negated = Lambda(lambda x: 1.0 - x)(qg_e)
    qx_prime_prime = layers.Dense(50, activation='tanh')(encoded_question)
    qx_hat_1 = Multiply()([qg_e, encoded_question])
    qx_hat_2 = Multiply()([qg_e_negated, qx_prime_prime])
    qx_hat = Add()([qx_hat_1, qx_hat_2])

    qwiq_w_b = layers.Input(shape=(query_maxlen, 2), dtype='float32')
    # qwiq_w_b = Lambda(lambda x: K.cast(x, dtype='float32'))(qwiq_w_b)

    qx_hat_wiq = layers.concatenate([qx_hat, qwiq_w_b])

    encoded_question = RNN(EMBED_HIDDEN_SIZE, return_sequences=True, kernel_regularizer=L1L2(l1=0.0, l2=0.0001))(qx_hat_wiq)
    # encoded_question = RNN(EMBED_HIDDEN_SIZE, return_sequences=True)(encoded_question)
    # encoded_question = layers.RepeatVector(story_maxlen)(encoded_question)
    alpha = layers.Dense(1)(encoded_question)
    alpha = layers.Reshape((query_maxlen,))(alpha)
    alpha = layers.Softmax()(alpha)

    z_hat = layers.Dot(axes=1)([encoded_question, alpha])

    """ Context proc """
    sentence = layers.Input(shape=(story_maxlen,), dtype='int32')
    emb_sentence = layers.Embedding(vocab_size, 50, trainable=False)(sentence)
    encoded_sentence = layers.Dropout(dropout_rate)(emb_sentence)
    g_e = layers.Dense(50, activation='sigmoid')(encoded_sentence)
    g_e_negated = Lambda(lambda x: 1.0 - x)(g_e)
    x_prime_prime = layers.Dense(50, activation='tanh')(encoded_sentence)
    x_hat_1 = Multiply()([g_e, encoded_sentence])
    x_hat_2 = Multiply()([g_e_negated, x_prime_prime])
    x_hat = Add()([x_hat_1, x_hat_2])

    emb_question_rep = RepeatVector4D(story_maxlen)(emb_question)
    emb_sentence_rep = RepeatVector4D_2(query_maxlen)(emb_sentence)
    similarity = Multiply()([emb_sentence_rep, emb_question_rep])
    similarity = layers.Dense(1)(similarity)
    similarity = layers.Reshape((story_maxlen, query_maxlen))(similarity)

    wiq_w = layers.Softmax()(similarity)
    wiq_w = Lambda(lambda x_in: K.sum(x_in, axis=-1))(wiq_w)
    wiq_w_reshp = layers.Reshape((story_maxlen, 1))(wiq_w)

    wiq_b = layers.Input(shape=(story_maxlen,), dtype='float32')
    wiq_b_reshp = layers.Reshape((story_maxlen, 1))(wiq_b)

    x_hat_wiq = layers.concatenate([x_hat, wiq_w_reshp, wiq_b_reshp])

    # merged = layers.add([encoded_sentence, encoded_question])
    # merged = layers.concatenate([x_hat_wiq, encoded_question])
    # merged = RNN(EMBED_HIDDEN_SIZE)(merged)
    h_sentence = RNN(EMBED_HIDDEN_SIZE, return_sequences=True, kernel_regularizer=L1L2(l1=0.0, l2=0.0001))(x_hat_wiq)
    # h_sentence = RNN(EMBED_HIDDEN_SIZE, return_sequences=True)(h_sentence)
    # h_sentence = layers.Dropout(dropout_rate)(h_sentence)

    z_hat_rep = layers.RepeatVector(story_maxlen)(z_hat)
    merged = layers.concatenate([h_sentence, z_hat_rep, layers.Multiply()([h_sentence, z_hat_rep])])
    s = layers.Dense(EMBED_HIDDEN_SIZE, activation='relu')(merged)

    # to_dense = layers.Dense(EMBED_HIDDEN_SIZE, activation='relu')(merged)
    # pred_span_start_idx = layers.Dense(span_max_idx, activation='softmax')(to_dense)

    pred_span_start_idx = layers.Dense(1)(s)
    pred_span_start_idx = layers.Reshape((span_max_idx,))(pred_span_start_idx)
    pred_span_start_idx = layers.Softmax()(pred_span_start_idx)

    model = Model([sentence, question, wiq_b, qwiq_w_b], pred_span_start_idx)
    return model


rnn = recurrent.GRU
model = build_FastQA_RNN_concat_model_GLOVE(rnn, 400, 20, 8000, 400, 0, 50, 0.2)


local_model_result_folder_path = '~/PycharmProjects/MemoryNet/model_results'
# local_model_result_folder_path = '/home/ec2-user/project/model_results'

m_saver = KerasS3ModelSaver(local_model_result_folder_path=local_model_result_folder_path)
s3_model_path, experiment_timestamp = m_saver.save_model(model=model, project_name='QA_QAngaroo', experiment_name='separate_files2',
                                                         protect_existing_folder=True)
print(s3_model_path)
print(experiment_timestamp)

_, y = make_classification(100)
y_pred = [random.randint(0, 1) for _ in range(100)]
class_pkg = ClassificationResultPackage()
class_pkg.prepare_result_package(y, y_pred, hyperparameters={'bash_script_path': 'dasdassd'},
                                 training_history=TrainingHistory({}, []))

res_saver = S3ResultsSaver(local_model_result_folder_path=local_model_result_folder_path)

s3_res_path, experiment_timestamp2 = res_saver.save_experiment_results(result_package=class_pkg,
                                                                       project_name='QA_QAngaroo', experiment_name='separate_files2',
                                                                       experiment_timestamp=experiment_timestamp,
                                                                       save_true_pred_labels=True,
                                                                       separate_files=True,
                                                                       protect_existing_folder=True)
