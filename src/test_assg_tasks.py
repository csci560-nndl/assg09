import numpy as np
import pandas as pd
import random
#import unittest
from twisted.trial import unittest
import tensorflow as tf
from tensorflow import keras
import keras_hub



class test_gru_music_generator_model(unittest.TestCase):

    def setUp(self):
        pass

    def test_model(self):
        num_dim = 42
        vocab_size = 123
        model = gru_music_generator_model(vocab_size, num_dim)

        self.assertEqual(len(model.layers), 3)

        l = model.layers[1]
        self.assertIsInstance(l, keras.layers.GRU)
        self.assertEqual(l.output.shape, (None, None, num_dim))
        self.assertTrue(l.return_sequences)

        l = model.layers[2]
        self.assertIsInstance(l, keras.layers.Dense)
        self.assertEqual(l.output.shape, (None, None, vocab_size))



class test_generate_jazz_solo(unittest.TestCase):

    def setUp(self):
        pass

    def test_given_example(self):
        vocab_size = 90
        num_dim = 32
        sequence_length = 30
        # create a model but we don't fit it, so will give kinda random output
        model = gru_music_generator_model(vocab_size, num_dim)
        solo = generate_jazz_solo(model, sequence_length, vocab_size, prompt=0, temperature=1.0)

        self.assertEqual(solo.shape, (sequence_length,))
        self.assertIsInstance(solo, np.ndarray)
        self.assertEqual(solo[0], 0)

    def test_sequence_vocab_shape(self):
        vocab_size = 256
        num_dim = 8
        sequence_length = 55
        # create a model but we don't fit it, so will give kinda random output
        model = gru_music_generator_model(vocab_size, num_dim)
        solo = generate_jazz_solo(model, sequence_length, vocab_size, prompt=5, temperature=3.0)

        self.assertEqual(solo.shape, (sequence_length,))
        self.assertIsInstance(solo, np.ndarray)
        self.assertEqual(solo[0], 5)


class test_transformer_music_generator_model(unittest.TestCase):

    def setUp(self):
        pass

    def test_model(self):
        num_dim = 42
        vocab_size = 123
        num_heads = 3
        model = transformer_music_generator_model(vocab_size, num_dim, num_heads)

        self.assertEqual(len(model.layers), 3)

        l = model.layers[1]
        self.assertIsInstance(l, keras_hub.layers.TransformerDecoder)
        self.assertEqual(l.output.shape, (None, None, vocab_size))
        self.assertEqual(l.num_heads, 3)

        l = model.layers[2]
        self.assertIsInstance(l, keras.layers.Dense)
        self.assertEqual(l.output.shape, (None, None, vocab_size))
