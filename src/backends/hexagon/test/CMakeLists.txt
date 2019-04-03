#
# Copyright © 2017 Arm Ltd. All rights reserved.
# Copyright © 2019 Linaro. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND hexagonBackendUnitTests_sources
	#HexagonCreateWorkloadTests.cpp
	#HexagonDetectionPostProcessTests.cpp
	#HexagonEndToEndTests.cpp
    HexagonJsonPrinterTests.cpp
    #HexagonLayerSupportTests.cpp
    #HexagonLayerTests.cpp
    #HexagonOptimizedNetworkTests.cpp
    #HexagonRuntimeTests.cpp
    #HexagonWorkloadFactoryHelper.hpp
)

add_library(hexagonBackendUnitTests OBJECT ${hexagonBackendUnitTests_sources})
target_include_directories(hexagonBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(hexagonBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(hexagonBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
