// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/cpu/encoder.h"

namespace mlx::core {

namespace {

template <typename T>
void arange(double start, double step, array& out, size_t size, Stream stream) {
  auto ptr = out.data<T>();
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_output_array(out);
  encoder.dispatch([ptr, start, step, size]() {
    if constexpr (std::is_integral_v<T>) {
      auto start_val = static_cast<T>(start);
      auto step_val = static_cast<T>(start + step) - start_val;
      for (size_t i = 0; i < size; ++i) {
        ptr[i] = start_val + static_cast<T>(i) * step_val;
      }
    } else if constexpr (std::is_same_v<T, complex64_t>) {
      for (size_t i = 0; i < size; ++i) {
        ptr[i] = static_cast<T>(
            static_cast<float>(start + static_cast<double>(i) * step));
      }
    } else {
      for (size_t i = 0; i < size; ++i) {
        ptr[i] = static_cast<T>(start + static_cast<double>(i) * step);
      }
    }
  });
}

} // namespace

} // namespace mlx::core
