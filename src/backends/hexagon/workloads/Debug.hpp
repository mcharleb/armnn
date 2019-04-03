//
// Copyright © 2017 Arm Ltd. All rights reserved.
// Copyright © 2019 Linaro. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Descriptors.hpp>
#include <armnn/Tensor.hpp>

namespace armnn
{
namespace hexagon
{

template <typename T>
void Debug(const TensorInfo& inputInfo,
           const TensorInfo& outputInfo,
           const DebugDescriptor& descriptor,
           const T* inputData,
           T* outputData);

} //namespace hexagon
} //namespace armnn
