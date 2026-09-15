cmake_minimum_required(VERSION 3.22)
project(luminal_hexagon_dsp LANGUAGES C)

if (NOT HEXAGON_SDK_ROOT)
    set(HEXAGON_SDK_ROOT $ENV{HEXAGON_SDK_ROOT})
endif()
if (NOT IS_DIRECTORY "${HEXAGON_SDK_ROOT}")
    message(FATAL_ERROR "Set HEXAGON_SDK_ROOT to the Hexagon SDK installation")
endif()

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)

set(DSP_VERSION v73 CACHE STRING "Hexagon architecture")
set(TARGET luminal-hexagon-${DSP_VERSION})

add_library(${TARGET} SHARED
    luminal_hexagon_dsp.c
    ${CMAKE_CURRENT_BINARY_DIR}/luminal_hexagon_skel.c
)
build_idl(luminal_hexagon.idl ${TARGET})

target_include_directories(${TARGET} PRIVATE
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${CMAKE_CURRENT_BINARY_DIR}
)
target_compile_options(${TARGET} PRIVATE -mv73 -mhvx -mhvx-length=128B)

install(TARGETS ${TARGET} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Windows-on-Snapdragon FastRPC requires the skel to be signed/catalogued.
# Keep this opt-in so ordinary cross compilation remains usable on Linux and
# simulator hosts.
if (HEXAGON_CERT)
    file(TO_CMAKE_PATH "$ENV{WINDOWS_SDK_BIN}/x86" WINDOWS_SDK_X86)
    file(TO_CMAKE_PATH "$ENV{WINDOWS_SDK_BIN}/arm64" WINDOWS_SDK_ARM64)
    find_program(INF2CAT NAMES Inf2Cat.exe inf2cat.exe PATHS ${WINDOWS_SDK_X86} REQUIRED)
    find_program(SIGNTOOL NAMES signtool.exe PATHS ${WINDOWS_SDK_ARM64} REQUIRED)

    set(SIGN_DIR ${CMAKE_CURRENT_BINARY_DIR}/sign)
    file(MAKE_DIRECTORY ${SIGN_DIR})
    configure_file(luminal_hexagon.inf.in ${SIGN_DIR}/luminal_hexagon.inf @ONLY)
    set(SIGN_COMMAND ${SIGNTOOL} sign /fd sha256 /f ${HEXAGON_CERT})
    if (HEXAGON_CERT_PASS)
        list(APPEND SIGN_COMMAND /p ${HEXAGON_CERT_PASS})
    endif()

    add_custom_target(luminal-hexagon-sign ALL
        DEPENDS ${TARGET}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:${TARGET}> ${SIGN_DIR}/libluminal-hexagon-${DSP_VERSION}.so
        COMMAND ${INF2CAT} /driver:${SIGN_DIR} /os:10_GE_ARM64
        COMMAND ${SIGN_COMMAND} ${SIGN_DIR}/libluminal-hexagon.cat
        COMMENT "Signing libluminal-hexagon-${DSP_VERSION}.so"
        VERBATIM
    )
endif()
