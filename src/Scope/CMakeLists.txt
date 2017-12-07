file(GLOB tmp *.cpp *.hpp)
set(scope_src ${scope_src} ${tmp})
set(scope_src ${scope_src} PARENT_SCOPE)

