//
// Copyright © 2017 Arm Ltd. All rights reserved.
// Copyright © 2019 Linaro. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Types.hpp>
#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include "HexagonStringMapping.hpp"

namespace armnn
{

template <typename Functor,
          typename armnn::DataType DataType,
          typename ParentDescriptor,
          typename armnn::hexagon::StringMapping::Id DebugString>
class HexagonComparisonWorkload
{
    // Needs specialization. The default is empty on purpose.
};

template <typename ParentDescriptor, typename Functor>
class HexagonUint8ComparisonWorkload : public BaseUint8ComparisonWorkload<ParentDescriptor>
{
public:
    using BaseUint8ComparisonWorkload<ParentDescriptor>::BaseUint8ComparisonWorkload;
    void ExecuteImpl(const char * debugString) const;
};

template <typename Functor, typename ParentDescriptor, typename armnn::hexagon::StringMapping::Id DebugString>
class HexagonComparisonWorkload<Functor, armnn::DataType::QuantisedAsymm8, ParentDescriptor, DebugString>
    : public HexagonUint8ComparisonWorkload<ParentDescriptor, Functor>
{
public:
    using HexagonUint8ComparisonWorkload<ParentDescriptor, Functor>::HexagonUint8ComparisonWorkload;

    virtual void Execute() const override
    {
        using Parent = HexagonUint8ComparisonWorkload<ParentDescriptor, Functor>;
        Parent::ExecuteImpl(hexagon::StringMapping::Instance().Get(DebugString));
    }
};

using HexagonEqualUint8Workload =
    HexagonComparisonWorkload<std::equal_to<uint8_t>,
                          DataType::QuantisedAsymm8,
                          EqualQueueDescriptor,
                          hexagon::StringMapping::HexagonEqualWorkload_Execute>;

using HexagonGreaterUint8Workload =
    HexagonComparisonWorkload<std::greater<uint8_t>,
                          DataType::QuantisedAsymm8,
                          GreaterQueueDescriptor,
                          hexagon::StringMapping::HexagonGreaterWorkload_Execute>;
} // armnn
