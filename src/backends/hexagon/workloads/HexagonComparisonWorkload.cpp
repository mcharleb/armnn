//
// Copyright © 2017 Arm Ltd. All rights reserved.
// Copyright © 2019 Linaro. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "HexagonComparisonWorkload.hpp"
#include "ElementwiseFunction.hpp"
#include "HexagonWorkloadUtils.hpp"
#include "Profiling.hpp"
#include "../HexagonBackend.hpp"
#include <vector>

namespace armnn {

template<typename ParentDescriptor, typename Functor>
void HexagonUint8ComparisonWorkload<ParentDescriptor, Functor>::ExecuteImpl(const char* debugString) const
{
    ARMNN_SCOPED_PROFILING_EVENT(HexagonBackend::GetIdStatic(), debugString);

    auto data = BaseUint8ComparisonWorkload<ParentDescriptor>::GetData();
    const TensorShape& inputInfo0 = GetTensorInfo(data.m_Inputs[0]).GetShape();
    const TensorShape& inputInfo1 = GetTensorInfo(data.m_Inputs[1]).GetShape();
    const TensorShape& outputShape = GetTensorInfo(data.m_Outputs[0]).GetShape();

    const uint8_t* inData0 = GetInputTensorData<uint8_t>(0, data);
    const uint8_t* inData1 = GetInputTensorData<uint8_t>(1, data);
    uint8_t* outData = GetOutputTensorData<uint8_t>(0, data);

    ElementwiseFunction<Functor, uint8_t, uint8_t>(inputInfo0,
                                                   inputInfo1,
                                                   outputShape,
                                                   inData0,
                                                   inData1,
                                                   outData);
}

}

template class armnn::HexagonUint8ComparisonWorkload<armnn::EqualQueueDescriptor, std::equal_to<uint8_t>>;

template class armnn::HexagonUint8ComparisonWorkload<armnn::GreaterQueueDescriptor, std::greater<uint8_t>>;
