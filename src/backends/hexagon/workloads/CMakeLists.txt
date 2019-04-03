#
# Copyright © 2017 Arm Ltd. All rights reserved.
# Copyright © 2019 Linaro. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND hexagonBackendWorkloads_sources
    Broadcast.cpp
    Broadcast.hpp
    Debug.cpp
    Debug.hpp
    ElementwiseFunction.cpp
    ElementwiseFunction.hpp
    HexagonComparisonWorkload.cpp
    HexagonComparisonWorkload.hpp
    HexagonDebugWorkload.cpp
    HexagonDebugWorkload.hpp
    HexagonElementwiseWorkload.cpp
    HexagonElementwiseWorkload.hpp
    HexagonWorkloads.hpp
    HexagonWorkloadUtils.hpp
    Minimum.hpp
    Maximum.hpp
    HexagonStringMapping.cpp
    HexagonStringMapping.hpp
)

add_library(hexagonBackendWorkloads OBJECT ${hexagonBackendWorkloads_sources})
target_include_directories(hexagonBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(hexagonBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(hexagonBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
