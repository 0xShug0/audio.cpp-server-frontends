set(AUDIOCPP_EXTERNAL_SERVER_FRONTENDS_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(audiocpp_external_frontend_requested AUDIOCPP_EXTERNAL_FRONTEND_NAME AUDIOCPP_EXTERNAL_FRONTEND_OUT)
    set(AUDIOCPP_EXTERNAL_FRONTEND_FOUND OFF)
    foreach(AUDIOCPP_EXTERNAL_FRONTEND_MODULE IN LISTS AUDIOCPP_SERVER_FRONTEND_MODULES)
        if (AUDIOCPP_EXTERNAL_FRONTEND_MODULE STREQUAL "${AUDIOCPP_EXTERNAL_FRONTEND_NAME}")
            set(AUDIOCPP_EXTERNAL_FRONTEND_FOUND ON)
        endif()
    endforeach()
    set(${AUDIOCPP_EXTERNAL_FRONTEND_OUT} ${AUDIOCPP_EXTERNAL_FRONTEND_FOUND} PARENT_SCOPE)
endfunction()

audiocpp_external_frontend_requested(audio_decode AUDIOCPP_EXTERNAL_FRONTEND_AUDIO_DECODE)
if (AUDIOCPP_EXTERNAL_FRONTEND_AUDIO_DECODE)
    audiocpp_register_server_frontend(audio_decode
        REGISTER_FUNCTION register_audio_decode_module
        SOURCES
            "${AUDIOCPP_EXTERNAL_SERVER_FRONTENDS_SOURCE_DIR}/modules/audio_decode.cpp"
        INCLUDE_DIRS
            "${AUDIOCPP_EXTERNAL_SERVER_FRONTENDS_SOURCE_DIR}/modules"
            "${AUDIOCPP_EXTERNAL_SERVER_FRONTENDS_SOURCE_DIR}/external/miniaudio")
endif()

audiocpp_external_frontend_requested(mp3_encode AUDIOCPP_EXTERNAL_FRONTEND_MP3_ENCODE)
if (AUDIOCPP_EXTERNAL_FRONTEND_MP3_ENCODE)
    set(AUDIOCPP_LAME_ROOT ""
        CACHE PATH "Optional libmp3lame install prefix for the server mp3_encode frontend")
    find_path(AUDIOCPP_LAME_INCLUDE_DIR
        NAMES lame/lame.h
        HINTS "${AUDIOCPP_LAME_ROOT}" "$ENV{CONDA_PREFIX}"
        PATH_SUFFIXES include Library/include)
    find_library(AUDIOCPP_LAME_LIBRARY
        NAMES mp3lame lame
        HINTS "${AUDIOCPP_LAME_ROOT}" "$ENV{CONDA_PREFIX}"
        PATH_SUFFIXES lib Library/lib)
    if (NOT AUDIOCPP_LAME_INCLUDE_DIR OR NOT AUDIOCPP_LAME_LIBRARY)
        message(FATAL_ERROR
            "AUDIOCPP_SERVER_FRONTEND_MODULES=mp3_encode requires libmp3lame headers and library "
            "(lame/lame.h and libmp3lame). Install libmp3lame with your package manager, "
            "use a conda environment that provides lame, or pass -DAUDIOCPP_LAME_ROOT=<prefix>.")
    endif()
    audiocpp_register_server_frontend(mp3_encode
        REGISTER_FUNCTION register_mp3_encode_module
        SOURCES
            "${AUDIOCPP_EXTERNAL_SERVER_FRONTENDS_SOURCE_DIR}/modules/mp3_encode.cpp"
        INCLUDE_DIRS
            "${AUDIOCPP_EXTERNAL_SERVER_FRONTENDS_SOURCE_DIR}/modules"
            "${AUDIOCPP_LAME_INCLUDE_DIR}"
        LIBRARIES
            "${AUDIOCPP_LAME_LIBRARY}")
endif()

audiocpp_external_frontend_requested(https AUDIOCPP_EXTERNAL_FRONTEND_HTTPS)
if (AUDIOCPP_EXTERNAL_FRONTEND_HTTPS)
    audiocpp_register_server_frontend(https
        SOURCES
            "${AUDIOCPP_EXTERNAL_SERVER_FRONTENDS_SOURCE_DIR}/modules/https.cpp"
        INCLUDE_DIRS
            "${AUDIOCPP_EXTERNAL_SERVER_FRONTENDS_SOURCE_DIR}/modules"
        LIBRARIES
            audiocpp_cpp_httplib
        COMPILE_DEFINITIONS
            AUDIOCPP_SERVER_FRONTEND_HAS_HTTPS=1)
endif()
