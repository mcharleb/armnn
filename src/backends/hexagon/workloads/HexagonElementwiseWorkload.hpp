//
// Copyright © 2017 Arm Ltd. All rights reserved.
// Copyright © 2019 Linaro. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Types.hpp>
#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include "Maximum.hpp"
#include "Minimum.hpp"
#include "HexagonStringMapping.hpp"

namespace armnn
{

template <typename Functor,
          typename armnn::DataType DataType,
          typename ParentDescriptor,
          typename armnn::hexagon::StringMapping::Id DebugString>
class HexagonElementwiseWorkload
{
    // Needs specialization. The default is empty on purpose.
};

template <typename ParentDescriptor, typename Functor>
class BaseUint8ElementwiseWorkload : public Uint8Workload<ParentDescriptor>
{
public:
    using Uint8Workload<ParentDescriptor>::Uint8Workload;
    void ExecuteImpl(const char * debugString) const;
};

template <typename Functor,
          typename ParentDescriptor,
          typename armnn::hexagon::StringMapping::Id DebugString>
class HexagonElementwiseWorkload<Functor, armnn::DataType::QuantisedAsymm8, ParentDescriptor, DebugString>
    : public BaseUint8ElementwiseWorkload<ParentDescriptor, Functor>
{
public:
    using BaseUint8ElementwiseWorkload<ParentDescriptor, Functor>::BaseUint8ElementwiseWorkload;

    virtual void Execute() const override
    {
        using Parent = BaseUint8ElementwiseWorkload<ParentDescriptor, Functor>;
        Parent::ExecuteImpl(hexagon::StringMapping::Instance().Get(DebugString));
    }
};

using HexagonAdditionUint8Workload =
    HexagonElementwiseWorkload<std::plus<float>,
                          DataType::QuantisedAsymm8,
                          AdditionQueueDescriptor,
                          hexagon::StringMapping::HexagonAdditionWorkload_Execute>;

using HexagonSubtractionUint8Workload =
    HexagonElementwiseWorkload<std::minus<float>,
                          DataType::QuantisedAsymm8,
                          SubtractionQueueDescriptor,
                          hexagon::StringMapping::HexagonSubtractionWorkload_Execute>;

using HexagonMultiplicationUint8Workload =
    HexagonElementwiseWorkload<std::multiplies<float>,
                          DataType::QuantisedAsymm8,
                          MultiplicationQueueDescriptor,
                          hexagon::StringMapping::HexagonMultiplicationWorkload_Execute>;

using HexagonDivisionUint8Workload =
    HexagonElementwiseWorkload<std::divides<float>,
                          DataType::QuantisedAsymm8,
                          DivisionQueueDescriptor,
                          hexagon::StringMapping::HexagonDivisionWorkload_Execute>;


using HexagonMaximumUint8Workload =
    HexagonElementwiseWorkload<hexagon::maximum<float>,
                          DataType::QuantisedAsymm8,
                          MaximumQueueDescriptor,
                          hexagon::StringMapping::HexagonMaximumWorkload_Execute>;

using HexagonMinimumUint8Workload =
    HexagonElementwiseWorkload<hexagon::minimum<float>,
                          DataType::QuantisedAsymm8,
                          MinimumQueueDescriptor,
                          hexagon::StringMapping::HexagonMinimumWorkload_Execute>;
} // armnn
