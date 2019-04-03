#
# Copyright © 2017 Arm Ltd. All rights reserved.
# Copyright © 2019 Linaro. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND hexagonBackend_sources
    HexagonBackend.cpp
    HexagonBackend.hpp
    HexagonBackendId.hpp
    HexagonLayerSupport.cpp
    HexagonLayerSupport.hpp
    HexagonWorkloadFactory.cpp
    HexagonWorkloadFactory.hpp

)

add_library(hexagonBackend OBJECT ${hexagonBackend_sources})
target_include_directories(hexagonBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(hexagonBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(hexagonBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)

add_subdirectory(workloads)

#if(BUILD_UNIT_TESTS)
#    add_subdirectory(test)
#endif()
