//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "ParserFlatbuffersFixture.hpp"
#include "../TfLiteParser.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowLiteParser)

struct ActivationFixture : ParserFlatbuffersFixture
{

    explicit ActivationFixture(std::string activationFunction, std::string dataType)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": )" + activationFunction + R"( } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": [ 1, 7 ],
                            "type": )" + dataType + R"(,
                            "buffer": 0,
                            "name": "inputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": [ 1, 7 ],
                            "type": )" + dataType + R"(,
                            "buffer": 1,
                            "name": "outputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        }
                    ],
                    "inputs": [ 0 ],
                    "outputs": [ 1 ],
                    "operators": [
                        {
                          "opcode_index": 0,
                          "inputs": [ 0 ],
                          "outputs": [ 1 ],
                          "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [ {}, {} ]
            }
        )";
        SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }

};

struct ReLuFixture : ActivationFixture
{
    ReLuFixture() : ActivationFixture("RELU", "FLOAT32") {}
};
BOOST_FIXTURE_TEST_CASE(ParseReLu, ReLuFixture)
{
    RunTest<2, armnn::DataType::Float32>(0, { -1.0f, -0.5f, 1.25f, -3.0f, 0.0f, 0.5f, -0.75f },
                                         { 0.0f, 0.0f, 1.25f, 0.0f, 0.0f, 0.5f, 0.0f });
}

struct ReLu6Fixture : ActivationFixture
{
    ReLu6Fixture() : ActivationFixture("RELU6", "FLOAT32") {}
};
BOOST_FIXTURE_TEST_CASE(ParseReLu6, ReLu6Fixture)
{
    RunTest<2, armnn::DataType::Float32>(0, { -1.0f, -0.5f, 7.25f, -3.0f, 0.0f, 0.5f, -0.75f },
                                         { 0.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.5f, 0.0f });
}

struct SigmoidFixture : ActivationFixture
{
    SigmoidFixture() : ActivationFixture("LOGISTIC", "FLOAT32") {}
};
BOOST_FIXTURE_TEST_CASE(ParseLogistic, SigmoidFixture)
{
    RunTest<2, armnn::DataType::Float32>(0, { -1.0f,     -0.5f,      4.0f,       -4.0f,  0.0f,      0.5f,     -0.75f },
                                         {0.268941f, 0.377541f, 0.982013f,  0.0179862f,  0.5f, 0.622459f,  0.320821f });
}
BOOST_AUTO_TEST_SUITE_END()
