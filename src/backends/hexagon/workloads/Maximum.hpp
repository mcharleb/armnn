//
// Copyright © 2017 Arm Ltd. All rights reserved.
// Copyright © 2019 Linaro. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <iostream>

namespace armnn
{
namespace hexagon
{
    template<typename T>
    struct maximum
    {
        T
        operator () (const T&  inputData0, const T&  inputData1) const
        {
            return std::max(inputData0, inputData1);
        }
    };

} //namespace hexagon
} //namespace armnn
