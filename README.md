multiscaler
===========

Decompose and recompose an image into a DCT pyramid.

Every layer of the pyramid contains as many DCT (low) frequencies as its number of pixels.
There is no smoothing, so ringing is to be expected. This code is intended to allow multiscale denoising of images.

Building
--------

To compile, use

    $ mkdir build
    $ cd build
    $ cmake .. [-D CMAKE_CXX_COMPILER=/path/of/c++/compiler -D CMAKE_C_COMPILER=/path/of/c/compiler] [-D CMAKE_BUILD_TYPE=Debug]
    $ make

To rebuild, e.g. when the code is modified, use

    $ cd build
    $ make

Using
-----

To decompose an image `input.png` into 3 levels, respectively `pyramid0.tiff`, `pyramid1.tiff` and `pyramid2.tiff`, use

    $ decompose input.png pyramid 3 .tiff

To recompose it (possibly after working on the single layers) on `output.tiff` use

    $ recompose pyramid 3 .tiff output.tiff