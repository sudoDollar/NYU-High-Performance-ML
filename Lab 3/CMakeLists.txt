cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(nonPython_chatbot nonPython_chatbot.cc)
target_link_libraries(nonPython_chatbot "${TORCH_LIBRARIES}")
set_property(TARGET nonPython_chatbot PROPERTY CXX_STANDARD 17)