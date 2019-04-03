//
// Copyright © 2019 Linaro. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <backendsCommon/IBackendInternal.hpp>

namespace armnn
{

class HexagonBackend : public IBackendInternal
{
public:
    HexagonBackend()  = default;
    ~HexagonBackend() = default;

    static const BackendId& GetIdStatic();
    const BackendId& GetId() const override { return GetIdStatic(); }

    IBackendInternal::IMemoryManagerUniquePtr CreateMemoryManager() const override;

    IBackendInternal::IWorkloadFactoryPtr CreateWorkloadFactory(
        const IBackendInternal::IMemoryManagerSharedPtr& memoryManager = nullptr) const override;

    IBackendInternal::IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const override;

    IBackendInternal::ISubGraphConverterPtr CreateSubGraphConverter(
        const std::shared_ptr<SubGraph>& subGraph) const override;

    IBackendInternal::Optimizations GetOptimizations() const override;
    IBackendInternal::ILayerSupportSharedPtr GetLayerSupport() const override;

    IBackendInternal::SubGraphUniquePtr OptimizeSubGraph(const SubGraph& subGraph,
                                                         bool& optimizationAttempted) const override;
};

} // namespace armnn
