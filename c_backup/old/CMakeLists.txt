cmake_minimum_required(VERSION 3.16)
project(lstm_mptemporal C)

# ── Opzioni ──────────────────────────────────────────────────────────────────
option(ENABLE_ASAN "Enable AddressSanitizer + UBSan in Debug builds" ON)
option(USE_FAST_MATH "Use inline approx (fast tanh/exp/sigmoid) for L0" ON)
option(USE_STATIC_WORKBUF "Use static work buffer in .bss" OFF)

# Parametri Mixed-Precision (puoi sovrascriverli da riga di comando)
set(MP_TAU_UP          "0.05f"  CACHE STRING "Up-threshold: enter L1")
set(MP_TAU_DOWN        "0.025f" CACHE STRING "Down-threshold: back to L0")
set(MP_COOLDOWN_STEPS  "8"      CACHE STRING "Min dwell time in L1")
set(MP_SENTINEL_K      "0"      CACHE STRING "0=off; >0 forces L1 every K steps")
set(MP_SOFTMAX_MODE    "1"      CACHE STRING "0=L0 fast exp; 1=L1 full exp")

# ── Standard e build type ────────────────────────────────────────────────────
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# ── Target ───────────────────────────────────────────────────────────────────
add_executable(lstm_demo
  main.c
  # lstm_model.h e vectors.h sono header-only e inclusi da main.c
)

# Warning utili (GCC/Clang)
if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(lstm_demo PRIVATE
    -Wall -Wextra -Wpedantic
    -Wshadow -Wpointer-arith -Wcast-qual
    -fno-omit-frame-pointer
  )
endif()

# Sanitizer in Debug (Address + Undefined)
if(ENABLE_ASAN AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(lstm_demo PRIVATE -O0 -g -fsanitize=address,undefined)
    target_link_options   (lstm_demo PRIVATE -fsanitize=address,undefined)
  endif()
endif()

# Definizioni per controllare il comportamento dal CMake
target_compile_definitions(lstm_demo PRIVATE
  $<$<BOOL:${USE_FAST_MATH}>:USE_FAST_MATH=1>
  $<$<BOOL:${USE_STATIC_WORKBUF}>:USE_STATIC_WORKBUF=1>
  MP_TAU_UP=${MP_TAU_UP}
  MP_TAU_DOWN=${MP_TAU_DOWN}
  MP_COOLDOWN_STEPS=${MP_COOLDOWN_STEPS}
  MP_SENTINEL_K=${MP_SENTINEL_K}
  MP_SOFTMAX_MODE=${MP_SOFTMAX_MODE}
)

# Link a libm (Linux)
target_link_libraries(lstm_demo m)
