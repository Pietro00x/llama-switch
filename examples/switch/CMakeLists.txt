set(TARGET llama-switch)

add_executable(${TARGET}
    main.cpp
    menu.cpp
    chat_screen.cpp
    download_screen.cpp
    ui.cpp
    util.cpp
    sysinfo.cpp
    models.cpp
    catalog.cpp
    engine.cpp
    generate.cpp
    download.cpp
    )

target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/vendor)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

nx_generate_nacp(${TARGET}.nacp
    NAME    "llama.cpp Switch"
    AUTHOR  "LLama-cpp-switch"
    VERSION "1.0.0"
    )

nx_create_nro(${TARGET}
    NACP ${TARGET}.nacp
    )
