//
// Copyright © 2019 Linaro. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "HexagonLayerSupport.hpp"
#include "HexagonBackendId.hpp"

#include <InternalTypes.hpp>
#include <LayerSupportCommon.hpp>
#include <armnn/Types.hpp>

#include <backendsCommon/BackendRegistry.hpp>

#include <boost/core/ignore_unused.hpp>

using namespace boost;

namespace armnn
{

namespace
{

template<typename Float32Func, typename Uint8Func, typename ... Params>
bool IsSupportedForDataTypeHexagon(Optional<std::string&> reasonIfUnsupported,
                               DataType dataType,
                               Float32Func floatFuncPtr,
                               Uint8Func uint8FuncPtr,
                               Params&&... params)
{
    return IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                         dataType,
                                         &FalseFunc<Params...>,
                                         floatFuncPtr,
                                         uint8FuncPtr,
                                         &FalseFunc<Params...>,
                                         &FalseFunc<Params...>,
                                         std::forward<Params>(params)...);
}

} // anonymous namespace

bool HexagonLayerSupport::IsAdditionSupported(const TensorInfo& input0,
                                          const TensorInfo& input1,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsSupportedForDataTypeHexagon(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &FalseFunc<>,
                                     &TrueFunc<>);
}

bool HexagonLayerSupport::IsDebugSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    return IsSupportedForDataTypeHexagon(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &FalseFunc<>,
                                     &TrueFunc<>);
}

bool HexagonLayerSupport::IsDivisionSupported(const TensorInfo& input0,
                                          const TensorInfo& input1,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsSupportedForDataTypeHexagon(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &FalseFunc<>,
                                     &TrueFunc<>);
}

bool HexagonLayerSupport::IsEqualSupported(const TensorInfo& input0,
                                       const TensorInfo& input1,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input0);
    ignore_unused(input1);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return IsSupportedForDataTypeHexagon(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &FalseFunc<>,
                                     &TrueFunc<>);
}

bool HexagonLayerSupport::IsFloorSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    return IsSupportedForDataTypeHexagon(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &FalseFunc<>,
                                     &TrueFunc<>);
}

bool HexagonLayerSupport::IsGreaterSupported(const TensorInfo& input0,
                                         const TensorInfo& input1,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input0);
    ignore_unused(input1);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return IsSupportedForDataTypeHexagon(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &FalseFunc<>,
                                     &TrueFunc<>);
}

bool HexagonLayerSupport::IsInputSupported(const TensorInfo& input,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    return IsSupportedForDataTypeHexagon(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &FalseFunc<>,
                                     &TrueFunc<>);
}

bool HexagonLayerSupport::IsMaximumSupported(const TensorInfo& input0,
                                         const TensorInfo& input1,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsSupportedForDataTypeHexagon(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &FalseFunc<>,
                                     &TrueFunc<>);
}

bool HexagonLayerSupport::IsMinimumSupported(const TensorInfo& input0,
                                         const TensorInfo& input1,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsSupportedForDataTypeHexagon(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &FalseFunc<>,
                                     &TrueFunc<>);
}

bool HexagonLayerSupport::IsMultiplicationSupported(const TensorInfo& input0,
                                                const TensorInfo& input1,
                                                const TensorInfo& output,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsSupportedForDataTypeHexagon(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &FalseFunc<>,
                                     &TrueFunc<>);
}

bool HexagonLayerSupport::IsOutputSupported(const TensorInfo& output,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return IsSupportedForDataTypeHexagon(reasonIfUnsupported,
                                         output.GetDataType(),
                                         &FalseFunc<>,
                                         &TrueFunc<>);
}

bool HexagonLayerSupport::IsSubtractionSupported(const TensorInfo& input0,
                                             const TensorInfo& input1,
                                             const TensorInfo& output,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsSupportedForDataTypeHexagon(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &FalseFunc<>,
                                     &TrueFunc<>);
}

} // namespace armnn
