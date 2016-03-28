#cross-compiling for Android
if(ANDROID)
    add_subdirectory(android)
#not cross-compiling
else()

    #helper
    function(add_isaac_test DIRECTORY NAME)
         add_executable(test-${NAME} ${DIRECTORY}/${NAME}.cpp)
         add_test(test-${NAME} test-${NAME})
         target_link_libraries(test-${NAME} isaac)
    endfunction()
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    #api
    foreach(NAME blas-1 blas-2 blas-3)
        add_isaac_test("api" ${NAME})
    endforeach()
    #compiler
    foreach(NAME element-1d element-2d reduce-1d reduce-2d)
        add_isaac_test("compiler" ${NAME})
    endforeach()
    #scheduler
    foreach(NAME heft)
        add_isaac_test("scheduler" ${NAME})
    endforeach()
endif()
