find_path(DXC_INCLUDE_DIR "dxcapi.h" PATH_SUFFIXES inc)
find_path(DXC_DLL_DIR "dxcompiler.dll" PATH_SUFFIXES bin/x64)
      
set(DXC_INCLUDE_DIRS ${DXC_INCLUDE_DIR})

FIND_LIBRARY(DXC_LIBRARY NAMES dxcompiler.lib PATH_SUFFIXES lib/x64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DXC DEFAULT_MSG DXC_INCLUDE_DIR DXC_DLL_DIR DXC_LIBRARY)

mark_as_advanced(DXC_INCLUDE_DIR DXC_DLL_DIR DXC_LIBRARY)
