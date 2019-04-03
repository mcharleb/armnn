//
// Copyright © 2017 Linaro. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "HexagonBackend.hpp"
#include "HexagonBackendId.hpp"
#include "HexagonWorkloadFactory.hpp"
#include "HexagonLayerSupport.hpp"

#include <backendsCommon/IBackendContext.hpp>
#include <backendsCommon/IMemoryManager.hpp>
#include <backendsCommon/BackendRegistry.hpp>

#include <Optimizer.hpp>

#include <boost/cast.hpp>

namespace armnn
{

namespace
{

static BackendRegistry::StaticRegistryInitializer g_RegisterHelper
{
    BackendRegistryInstance(),
    HexagonBackend::GetIdStatic(),
    []()
    {
        return IBackendInternalUniquePtr(new HexagonBackend);
    }
};

}

const BackendId& HexagonBackend::GetIdStatic()
{
    static const BackendId s_Id{HexagonBackendId()};
    return s_Id;
}

IBackendInternal::IWorkloadFactoryPtr HexagonBackend::CreateWorkloadFactory(
    const IBackendInternal::IMemoryManagerSharedPtr& memoryManager) const
{
    return std::make_unique<HexagonWorkloadFactory>();
}

IBackendInternal::IBackendContextPtr HexagonBackend::CreateBackendContext(const IRuntime::CreationOptions&) const
{
    return IBackendContextPtr{};
}

IBackendInternal::IMemoryManagerUniquePtr HexagonBackend::CreateMemoryManager() const
{
    return IMemoryManagerUniquePtr{};
}

IBackendInternal::ISubGraphConverterPtr HexagonBackend::CreateSubGraphConverter(
    const std::shared_ptr<SubGraph>& subGraph) const
{
    return ISubGraphConverterPtr{};
}

IBackendInternal::Optimizations HexagonBackend::GetOptimizations() const
{
    return Optimizations{};
}

IBackendInternal::ILayerSupportSharedPtr HexagonBackend::GetLayerSupport() const
{
    static ILayerSupportSharedPtr layerSupport{new HexagonLayerSupport};
    return layerSupport;
}

IBackendInternal::SubGraphUniquePtr HexagonBackend::OptimizeSubGraph(const SubGraph& subGraph,
                                                                 bool& optimizationAttempted) const
{
    // Not trying to optimize the given sub-graph
    optimizationAttempted = false;

    return SubGraphUniquePtr{};
}

} // namespace armnn
