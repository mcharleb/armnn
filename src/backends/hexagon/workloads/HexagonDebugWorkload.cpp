//
// Copyright © 2017 Arm Ltd. All rights reserved.
// Copyright © 2019 Linaro. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "HexagonDebugWorkload.hpp"
#include "Debug.hpp"
#include "HexagonWorkloadUtils.hpp"
#include "../HexagonBackend.hpp"

#include <TypeUtils.hpp>

namespace armnn
{

template<armnn::DataType DataType>
void HexagonDebugWorkload<DataType>::Execute() const
{
    using T = ResolveType<DataType>;

    ARMNN_SCOPED_PROFILING_EVENT(HexagonBackend::GetIdStatic(), GetName() + "_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    const T* inputData = GetInputTensorData<T>(0, m_Data);
    T* outputData = GetOutputTensorData<T>(0, m_Data);

    hexagon::Debug(inputInfo, outputInfo, m_Data.m_Parameters, inputData, outputData);
}

template class HexagonDebugWorkload<DataType::QuantisedAsymm8>;

} // namespace armnn
