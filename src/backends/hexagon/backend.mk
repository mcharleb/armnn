#
# Copyright © 2019 Linaro. All rights reserved.
# SPDX-License-Identifier: MIT
#

# BACKEND_SOURCES contains the list of files to be included
# in the Android build and it is picked up by the Android.mk
# file in the root of ArmNN

BACKEND_SOURCES := \
        HexagonBackend.cpp \
        HexagonLayerSupport.cpp \
        HexagonWorkloadFactory.cpp \
        workloads/Debug.cpp \
        workloads/ElementwiseFunction.cpp \

# BACKEND_TEST_SOURCES contains the list of files to be included
# in the Android unit test build (armnn-tests) and it is picked
# up by the Android.mk file in the root of ArmNN

BACKEND_TEST_SOURCES :=
#\
#        test/HexagonCreateWorkloadTests.cpp \
#        test/HexagonEndToEndTests.cpp \
#        test/HexagonJsonPrinterTests.cpp \
#        test/HexagonLayerSupportTests.cpp \
#        test/HexagonLayerTests.cpp \
#        test/HexagonOptimizedNetworkTests.cpp \
#        test/HexagonRuntimeTests.cpp
