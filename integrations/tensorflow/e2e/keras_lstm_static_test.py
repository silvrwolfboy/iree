# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This test is the same as keras_lstm_test, but all shapes are static.
# This stresses the TensorList lowering more specifically.

from pyiree.tf.support import tf_test_utils
import tensorflow.compat.v2 as tf

NUM_UNITS = 10
NUM_TIMESTEPS = 24
NUM_BATCH = 7


class Lstm(tf.Module):

  def __init__(self):
    super(Lstm, self).__init__()
    self.lstm = tf.keras.layers.LSTM(units=NUM_UNITS, return_sequences=True)

  @tf.function(input_signature=[
      tf.TensorSpec([NUM_BATCH, NUM_TIMESTEPS, NUM_UNITS], tf.float32)
  ])
  def predict(self, x):
    return self.lstm(x)


# TODO(silvasean): Get this test working on IREE.
@tf_test_utils.compile_modules(backends=["tf"], lstm=(Lstm, ["predict"]))
class LstmTest(tf_test_utils.SavedModelTestCase):

  def test_lstm(self):
    m = self.modules.lstm.all
    m.predict(tf.constant(0., shape=[NUM_BATCH, NUM_TIMESTEPS,
                                     NUM_UNITS])).print().assert_all_close()


if __name__ == "__main__":
  if hasattr(tf, "enable_v2_behavior"):
    tf.enable_v2_behavior()
  tf.test.main()
