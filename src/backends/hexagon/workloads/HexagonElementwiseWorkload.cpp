//
// Copyright © 2017 Arm Ltd. All rights reserved.
// Copyright © 2019 Linaro. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "HexagonElementwiseWorkload.hpp"
#include "ElementwiseFunction.hpp"
#include "HexagonWorkloadUtils.hpp"
#include "Profiling.hpp"
#include "../HexagonBackend.hpp"
#include <vector>

namespace armnn
{

template <typename ParentDescriptor, typename Functor>
void BaseUint8ElementwiseWorkload<ParentDescriptor, Functor>::ExecuteImpl(const char * debugString) const
{
    ARMNN_SCOPED_PROFILING_EVENT(HexagonBackend::GetIdStatic(), debugString);

    auto data = Uint8Workload<ParentDescriptor>::GetData();
    const TensorInfo& inputInfo0 = GetTensorInfo(data.m_Inputs[0]);
    const TensorInfo& inputInfo1 = GetTensorInfo(data.m_Inputs[1]);
    const TensorInfo& outputInfo = GetTensorInfo(data.m_Outputs[0]);

    auto dequant0 = Dequantize(GetInputTensorDataU8(0, data), inputInfo0);
    auto dequant1 = Dequantize(GetInputTensorDataU8(1, data), inputInfo1);

    std::vector<float> results(outputInfo.GetNumElements());

    ElementwiseFunction<Functor, float, float>(inputInfo0.GetShape(),
                                               inputInfo1.GetShape(),
                                               outputInfo.GetShape(),
                                               dequant0.data(),
                                               dequant1.data(),
                                               results.data());

    Quantize(GetOutputTensorDataU8(0, data), results.data(), outputInfo);
}

}

template class armnn::BaseUint8ElementwiseWorkload<armnn::AdditionQueueDescriptor, std::plus<float>>;

template class armnn::BaseUint8ElementwiseWorkload<armnn::SubtractionQueueDescriptor, std::minus<float>>;

template class armnn::BaseUint8ElementwiseWorkload<armnn::MultiplicationQueueDescriptor, std::multiplies<float>>;

template class armnn::BaseUint8ElementwiseWorkload<armnn::DivisionQueueDescriptor, std::divides<float>>;

template class armnn::BaseUint8ElementwiseWorkload<armnn::MaximumQueueDescriptor, armnn::hexagon::maximum<float>>;

template class armnn::BaseUint8ElementwiseWorkload<armnn::MinimumQueueDescriptor, armnn::hexagon::minimum<float>>;
