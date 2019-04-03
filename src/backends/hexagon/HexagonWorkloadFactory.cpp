//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/MemCopyWorkload.hpp>
#include <backendsCommon/MakeWorkloadHelper.hpp>
#include "HexagonWorkloadFactory.hpp"
#include "HexagonBackendId.hpp"
#include "workloads/HexagonWorkloads.hpp"
#include "Layer.hpp"

#include <boost/log/trivial.hpp>

namespace armnn
{

namespace
{
static const BackendId s_Id{HexagonBackendId()};
}

template <typename F32Workload, typename U8Workload, typename QueueDescriptorType>
std::unique_ptr<IWorkload> HexagonWorkloadFactory::MakeWorkload(const QueueDescriptorType& descriptor,
    const WorkloadInfo& info) const
{
    return armnn::MakeWorkloadHelper<NullWorkload, F32Workload, U8Workload, NullWorkload, NullWorkload>(descriptor,
                                                                                                        info);
}

HexagonWorkloadFactory::HexagonWorkloadFactory()
{
}

const BackendId& HexagonWorkloadFactory::GetBackendId() const
{
    return s_Id;
}

bool HexagonWorkloadFactory::IsLayerSupported(const Layer& layer,
                                          Optional<DataType> dataType,
                                          std::string& outReasonIfUnsupported)
{
    return IWorkloadFactory::IsLayerSupported(s_Id, layer, dataType, outReasonIfUnsupported);
}

std::unique_ptr<ITensorHandle> HexagonWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo) const
{
    return std::make_unique<ScopedCpuTensorHandle>(tensorInfo);
}

std::unique_ptr<ITensorHandle> HexagonWorkloadFactory::CreateTensorHandle(const TensorInfo& tensorInfo,
                                                                      DataLayout dataLayout) const
{
    return std::make_unique<ScopedCpuTensorHandle>(tensorInfo);
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateInput(const InputQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    if (info.m_InputTensorInfos.empty() )
    {
        throw InvalidArgumentException("HexagonWorkloadFactory::CreateInput: Input cannot be zero length");
    }
    if (info.m_OutputTensorInfos.empty())
    {
        throw InvalidArgumentException("HexagonWorkloadFactory::CreateInput: Output cannot be zero length");
    }

    if (info.m_InputTensorInfos[0].GetNumBytes() != info.m_OutputTensorInfos[0].GetNumBytes())
    {
        throw InvalidArgumentException("HexagonWorkloadFactory::CreateInput: data input and output differ in byte count.");
    }

    return MakeWorkload<CopyMemGenericWorkload, CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateOutput(const OutputQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
    if (info.m_InputTensorInfos.empty() )
    {
        throw InvalidArgumentException("HexagonWorkloadFactory::CreateOutput: Input cannot be zero length");
    }
    if (info.m_OutputTensorInfos.empty())
    {
        throw InvalidArgumentException("HexagonWorkloadFactory::CreateOutput: Output cannot be zero length");
    }
    if (info.m_InputTensorInfos[0].GetNumBytes() != info.m_OutputTensorInfos[0].GetNumBytes())
    {
        throw InvalidArgumentException("HexagonWorkloadFactory::CreateOutput: data input and output differ in byte count.");
    }

    return MakeWorkloadHelper<CopyMemGenericWorkload, CopyMemGenericWorkload,
                              CopyMemGenericWorkload, NullWorkload, CopyMemGenericWorkload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> HexagonWorkloadFactory::CreateAddition(const AdditionQueueDescriptor& descriptor,
                                                                     const WorkloadInfo&            info) const
{
    return MakeWorkload<NullWorkload, HexagonAdditionUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> HexagonWorkloadFactory::CreateMultiplication(
    const MultiplicationQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, HexagonMultiplicationUint8Workload>(descriptor, info);
}


std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateFloor(const FloorQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, NullWorkload>(descriptor, info);
}


std::unique_ptr<armnn::IWorkload> HexagonWorkloadFactory::CreateDivision(
    const DivisionQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, HexagonDivisionUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> HexagonWorkloadFactory::CreateSubtraction(
    const SubtractionQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, HexagonSubtractionUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> HexagonWorkloadFactory::CreateMaximum(
    const MaximumQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, HexagonMaximumUint8Workload>(descriptor, info);
}

std::unique_ptr<armnn::IWorkload> HexagonWorkloadFactory::CreateMinimum(
    const MinimumQueueDescriptor& descriptor, const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, HexagonMinimumUint8Workload>(descriptor, info);
}


std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateEqual(const EqualQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, HexagonEqualUint8Workload>(descriptor, info);
}


std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateGreater(const GreaterQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, HexagonGreaterUint8Workload>(descriptor, info);
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateDebug(const DebugQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info) const
{
    return MakeWorkload<NullWorkload, HexagonDebugUint8Workload>(descriptor, info);
}

// Stubs
std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateActivation(const ActivationQueueDescriptor& descriptor,
                                            const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateSoftmax(const SoftmaxQueueDescriptor& descriptor,
                                             const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateSplitter(const SplitterQueueDescriptor& descriptor,
                                              const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateMerger(const MergerQueueDescriptor& descriptor,
                                            const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateFullyConnected(const FullyConnectedQueueDescriptor& descriptor,
                                                    const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreatePooling2d(const Pooling2dQueueDescriptor& descriptor,
                                               const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreatePermute(const PermuteQueueDescriptor& descriptor,
                                             const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateConvolution2d(const Convolution2dQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateDepthwiseConvolution2d(const DepthwiseConvolution2dQueueDescriptor& descriptor,
                                                            const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateDetectionPostProcess(const DetectionPostProcessQueueDescriptor& descriptor,
                                                          const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateNormalization(const NormalizationQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateBatchNormalization(const BatchNormalizationQueueDescriptor& descriptor,
                                                        const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateMemCopy(const MemCopyQueueDescriptor& descriptor,
                                             const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateResizeBilinear(const ResizeBilinearQueueDescriptor& descriptor,
                                                    const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateFakeQuantization(const FakeQuantizationQueueDescriptor& descriptor,
                                                      const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateL2Normalization(const L2NormalizationQueueDescriptor& descriptor,
                                                     const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateConstant(const ConstantQueueDescriptor& descriptor,
                                              const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateReshape(const ReshapeQueueDescriptor& descriptor,
                                             const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateSpaceToBatchNd(const SpaceToBatchNdQueueDescriptor& descriptor,
                                                    const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateLstm(const LstmQueueDescriptor& descriptor,
                                          const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateConvertFp16ToFp32(const ConvertFp16ToFp32QueueDescriptor& descriptor,
                                                       const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateConvertFp32ToFp16(const ConvertFp32ToFp16QueueDescriptor& descriptor,
                                                       const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateMean(const MeanQueueDescriptor& descriptor,
                                          const WorkloadInfo& Info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreatePad(const PadQueueDescriptor& descriptor,
                                         const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateBatchToSpaceNd(const BatchToSpaceNdQueueDescriptor& descriptor,
                                                    const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateStridedSlice(const StridedSliceQueueDescriptor& descriptor,
                                                  const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateRsqrt(const RsqrtQueueDescriptor& descriptor,
                                           const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreatePreCompiled(const PreCompiledQueueDescriptor& descriptor,
                                                 const WorkloadInfo& info) const
{
	return nullptr;
}

std::unique_ptr<IWorkload> HexagonWorkloadFactory::CreateGather(const GatherQueueDescriptor& descriptor,
                                            const WorkloadInfo& info) const
{
	return nullptr;
}


} // namespace armnn
