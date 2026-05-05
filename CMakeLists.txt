cmake_minimum_required(VERSION 3.10)
project(GameEngine)

# تعيين معايير C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# إضافة دعم Android NDK
find_library(android-lib android)
find_library(egl-lib EGL)
find_library(glesv3-lib GLESv3)
find_library(log-lib log)

# إضافة الملفات المصدرية
add_library(game_engine SHARED
    main.cpp
)

# ربط المكتبات
target_link_libraries(game_engine
    ${android-lib}
    ${egl-lib}
    ${glesv3-lib}
    ${log-lib}
    android_native_app_glue
)

# إعدادات الترجمة
target_compile_options(game_engine PRIVATE
    -Wall
    -Wextra
    -O2
    -funroll-loops
)