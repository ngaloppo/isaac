function(add_isaac_test DIRECTORY NAME)
     add_executable(test-${NAME} ${DIRECTORY}/${NAME}.cpp)
     #set_target_properties(test-${NAME} PROPERTIES OUTPUT_NAME ${NAME})
     add_test(test-${NAME} test-${NAME})
     target_link_libraries(test-${NAME} isaac)
endfunction()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#Operations
foreach(NAME element-wise-1d element-wise-2d reduce-1d reduce-2d matrix-multiply)
    add_isaac_test("operations" ${NAME})
endforeach()
#Scheduling
foreach(NAME heft)
    add_isaac_test("scheduling" ${NAME})
endforeach()
