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

#include <cstring>

namespace armnn
{


template<armnn::DataType DataType>
void HexagonDebugWorkload<DataType>::Execute() const
{
    using T = ResolveType<DataType>;

    ARMNN_SCOPED_PROFILING_EVENT(HexagonBackend::GetIdStatic(), GetName() + "_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);

    const T* inputData = GetInputTensorData<T>(0, m_Data);
    T* outputData = GetOutputTensorData<T>(0, m_Data);

    if (m_Callback)
    {
        m_Callback(m_Data.m_Guid, m_Data.m_SlotIndex, m_Data.m_Inputs[0]);
    }
    else
    {
	    hexagon::Debug(inputInfo, inputData, m_Data.m_Guid, m_Data.m_LayerName, m_Data.m_SlotIndex);
    }

    std::memcpy(outputData, inputData, inputInfo.GetNumElements()*sizeof(T));

}

template<armnn::DataType DataType>
void HexagonDebugWorkload<DataType>::RegisterDebugCallback(const DebugCallbackFunction& func)
{
    m_Callback = func;
}

template class HexagonDebugWorkload<DataType::QuantisedAsymm8>;
} // namespace armnn
