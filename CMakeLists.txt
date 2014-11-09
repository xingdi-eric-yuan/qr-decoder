project( qrdecoder )
find_package( OpenCV REQUIRED )
add_executable( qrdecoder qrdecoder )
target_link_libraries( qrdecoder ${OpenCV_LIBS} )