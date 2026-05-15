// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_exp.h"
#include "sfpi.h"

namespace ckernel::sfpu
{

template <bool APPROXIMATION_MODE /*unused*/, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void _calculate_exp2_()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];

        sfpi::vFloat result;

        if constexpr (is_fp32_dest_acc_en)
        {
            // exp2(x) = 2^x = exp(x * ln(2)).
            // The exp function internally rescales by 1/ln(2), so passing v directly
            // (instead of v * ln(2)) eliminates the unnecessary multiply by ln(2).
            result = _sfpu_exp_fp32_accurate_(v);
        }
        else
        {
            // For bf16, _sfpu_exp_21f_bf16_ already multiplies by 1/ln(2) internally.
            // Calling it directly with v (instead of v * ln(2)) removes the wasted
            // multiply and avoids the double-rounding through ln(2).
            result = _sfpu_exp_21f_bf16_<true>(v);
            result = sfpi::float_to_fp16b(result, sfpi::RoundMode::NearestEven);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE /*unused*/>
inline void _init_exp2_()
{
    // No longer needed — we no longer multiply by ln(2) before exp.
}

} // namespace ckernel::sfpu
