//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "HexagonStringMapping.hpp"

namespace armnn
{
namespace hexagon
{

const StringMapping& StringMapping::Instance()
{
    static StringMapping instance;
    return instance;
}

} // hexagon
} // armnn
