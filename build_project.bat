set currrentDir=%CD%
cd vsbuild
cmake --build . --target INSTALL --config Release
cd %currrentDir%
