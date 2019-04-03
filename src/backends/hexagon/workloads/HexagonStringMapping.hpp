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

///
/// StringMapping is helper class to be able to use strings as template
/// parameters, so this allows simplifying code which only differs in
/// a string, such as a debug string literal.
///
struct StringMapping
{
public:
    enum Id {
        HexagonAdditionWorkload_Execute,
        HexagonEqualWorkload_Execute,
        HexagonDivisionWorkload_Execute,
        HexagonGreaterWorkload_Execute,
        HexagonMaximumWorkload_Execute,
        HexagonMinimumWorkload_Execute,
        HexagonMultiplicationWorkload_Execute,
        HexagonSubtractionWorkload_Execute,
        MAX_STRING_ID
    };

    const char * Get(Id id) const
    {
        return m_Strings[id];
    }

    static const StringMapping& Instance();

private:
    StringMapping()
    {
        m_Strings[HexagonAdditionWorkload_Execute] = "HexagonAdditionWorkload_Execute";
        m_Strings[HexagonDivisionWorkload_Execute] = "HexagonDivisionWorkload_Execute";
        m_Strings[HexagonEqualWorkload_Execute] = "HexagonEqualWorkload_Execute";
        m_Strings[HexagonGreaterWorkload_Execute] = "HexagonGreaterWorkload_Execute";
        m_Strings[HexagonMaximumWorkload_Execute] = "HexagonMaximumWorkload_Execute";
        m_Strings[HexagonMinimumWorkload_Execute] = "HexagonMinimumWorkload_Execute";
        m_Strings[HexagonMultiplicationWorkload_Execute] = "HexagonMultiplicationWorkload_Execute";
        m_Strings[HexagonSubtractionWorkload_Execute] = "HexagonSubtractionWorkload_Execute";
    }

    StringMapping(const StringMapping &) = delete;
    StringMapping& operator=(const StringMapping &) = delete;

    const char * m_Strings[MAX_STRING_ID];
};
};

} //namespace armnn
