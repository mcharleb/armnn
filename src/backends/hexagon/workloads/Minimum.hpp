//
// Copyright © 2017 Arm Ltd. All rights reserved.
// Copyright © 2019 Linaro. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace armnn
{
namespace hexagon
{

template<typename T>
struct minimum
{
    T
    operator()(const T& input1, const T& input2) const
    {
        return std::min(input1, input2);
    }
};

} //namespace hexagon
} //namespace armnn

