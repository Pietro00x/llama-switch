set(TARGET llama-switch)
set(LLAMA_SWITCH_VERSION "1.0.0")

add_executable(${TARGET}
    main.cpp
    menu.cpp
    chat_screen.cpp
    download_screen.cpp
    bench_screen.cpp
    bench.cpp
    ui.cpp
    util.cpp
    sysinfo.cpp
    models.cpp
    catalog.cpp
    engine.cpp
    generate.cpp
    download.cpp
    clocks.cpp
    clocks_switch.cpp
    settings.cpp
    )

target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/vendor)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# version shown in (and saved with) the benchmark results; commit/build from cmake/build-info.cmake at
# configure time (scripts/build-switch.sh configures on every run). Only this file is rebuilt when they change.
set_source_files_properties(bench_screen.cpp PROPERTIES COMPILE_DEFINITIONS
    "LLAMA_SWITCH_VERSION=\"${LLAMA_SWITCH_VERSION}\";LLAMA_SWITCH_COMMIT=\"${LLAMA_BUILD_COMMIT}\";LLAMA_SWITCH_BUILD=${LLAMA_BUILD_NUMBER}")

nx_generate_nacp(${TARGET}.nacp
    NAME    "llama.cpp Switch"
    AUTHOR  "LLama-cpp-switch"
    VERSION "${LLAMA_SWITCH_VERSION}"
    )

nx_create_nro(${TARGET}
    NACP ${TARGET}.nacp
    )
