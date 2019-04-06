//
// Copyright © 2017 Arm Ltd. All rights reserved.
// Copyright © 2019 Linaro. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/TypesUtils.hpp>

#include <backendsCommon/Workload.hpp>

namespace armnn
{

template <armnn::DataType DataType>
class HexagonDebugWorkload : public TypedWorkload<DebugQueueDescriptor, DataType>
{
public:
    HexagonDebugWorkload(const DebugQueueDescriptor& descriptor, const WorkloadInfo& info)
    : TypedWorkload<DebugQueueDescriptor, DataType>(descriptor, info)
    , m_Callback(nullptr) {}

    static const std::string& GetName()
    {
        static const std::string name = std::string("HexagonDebug") + GetDataTypeName(DataType) + "Workload";
        return name;
    }

    using TypedWorkload<DebugQueueDescriptor, DataType>::m_Data;
    using TypedWorkload<DebugQueueDescriptor, DataType>::TypedWorkload;

    void Execute() const override;

    void RegisterDebugCallback(const DebugCallbackFunction& func) override;

private:
    DebugCallbackFunction m_Callback;
};

using HexagonDebugUint8Workload = HexagonDebugWorkload<DataType::QuantisedAsymm8>;

} // namespace armnn
