#
# Copyright © 2017 Arm Ltd. All rights reserved.
# Copyright © 2019 Linaro. All rights reserved.
# SPDX-License-Identifier: MIT
#

add_subdirectory(${PROJECT_SOURCE_DIR}/src/backends/hexagon)
list(APPEND armnnLibraries hexagonBackend hexagonBackendWorkloads)
#list(APPEND armnnUnitTestLibraries hexagonBackendUnitTests)
