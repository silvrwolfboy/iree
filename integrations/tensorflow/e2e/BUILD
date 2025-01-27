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

load(
    "//bindings/python:build_defs.oss.bzl",
    "INTREE_TENSORFLOW_PY_DEPS",
    "NUMPY_DEPS",
    "iree_py_test",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

[
    iree_py_test(
        name = name,
        srcs = [name + ".py"],
        python_version = "PY3",
        # TODO(b/145815906) Get this running in OSS CI.
        tags = ["nokokoro"],
        deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
            "//integrations/tensorflow/bindings/python/pyiree/tf/support",
        ],
    )
    for name in [
        "batch_norm_test",
        "control_flow_test",
        "dynamic_mlp_test",
        "exported_names_test",
        "tensorlist_test",
        "keras_lstm_test",
        "keras_lstm_static_test",
        "linspace_test",
        "mandelbrot_test",
        "math_test",
        "matrix_ops_test",
        "ring_buffer_test",
        "sliding_window_test",
        "simple_arithmetic_test",
        "simple_stateful_test",
    ]
]

[
    iree_py_test(
        name = "keras_vision_model_" + model_name + "_test",
        srcs = ["keras_vision_model_test.py"],
        args = [
            "--model=%s" % model_name,
            "--target_backends=%s" % backends,
        ],
        main = "keras_vision_model_test.py",
        python_version = "PY3",
        # TODO(b/145815906) Get this running in OSS CI.
        tags = [
            "large",
            "manual",
            "nokokoro",
        ],
        deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
            "//integrations/tensorflow/bindings/python/pyiree/tf/support",
        ],
    )
    # TODO(b/150244105) update backends with "tf,iree_interpreter,iree_vulkan"
    # and add other supported vision models
    for model_name, backends in [
        ("ResNet50", "tf"),
        ("MobileNet", "tf"),
        ("MobileNetV2", "tf"),
    ]
]

iree_py_test(
    name = "vulkan_conv_test",
    srcs = ["vulkan_conv_test.py"],
    python_version = "PY3",
    # TODO(b/145815906) Get this running in OSS CI.
    tags = [
        "noga",
        "nokokoro",
    ],
    deps = INTREE_TENSORFLOW_PY_DEPS + NUMPY_DEPS + [
        "//integrations/tensorflow/bindings/python/pyiree/tf/support",
    ],
)
