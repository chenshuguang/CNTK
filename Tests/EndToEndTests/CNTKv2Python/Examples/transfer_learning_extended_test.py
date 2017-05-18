# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import pytest
import sys
import json
from cntk import load_model
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import try_set_default_device, gpu
from cntk.logging.graph import get_node_outputs
from cntk.ops.tests.ops_test_utils import cntk_device

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "TransferLearning"))
from prepare_test_data import prepare_animals_data_unzipped
from TransferLearning_Extended import train_and_eval

TOLERANCE_ABSOLUTE = 2E-2

def test_transfer_learning(device_id):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU') # due to batch normalization in ResNet_18
    try_set_default_device(cntk_device(device_id))

    base_path = os.path.dirname(os.path.abspath(__file__))
    externalData = 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY' in os.environ
    if externalData:
        extPath = os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY']
        print("Reading data and model from %s" % extPath)
        model_file = os.path.join(extPath, *"PreTrainedModels/ResNet/v1/ResNet_18.model".split("/"))
    else:
        model_file = os.path.join(base_path, *"../../../../Examples/Image/PretrainedModels/ResNet_18.model".split("/"))

    animals_path = prepare_animals_data_unzipped(os.path.join(extPath, "Image"))
    train_image_folder = os.path.join(animals_path, "Train")
    test_image_folder = os.path.join(animals_path, "Test")
    output_file = os.path.join(base_path, "tl_extended_output.txt")

    train_and_eval(model_file, train_image_folder, test_image_folder, output_file, None, testing=True)

    expected_output_file = os.path.join(base_path, "tl_extended_expected_output.txt")

    with open(output_file) as output_json:
        output_lines = output_json.readlines()
    with open(expected_output_file) as expected_output_json:
        expected_output_lines = expected_output_json.readlines()

    for i in range(len(output_lines)):
        output = json.loads(output_lines[i])[0]
        expected_output = json.loads(expected_output_lines[i])[0]

        assert np.allclose(output["predictions"]["Sheep"], expected_output["predictions"]["Sheep"], atol=TOLERANCE_ABSOLUTE)
        assert np.allclose(output["predictions"]["Wolf"], expected_output["predictions"]["Wolf"], atol=TOLERANCE_ABSOLUTE)
