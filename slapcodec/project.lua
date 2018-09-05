ProjectName = "slapcodec"
project(ProjectName)

  --Settings
  kind "StaticLib"
  language "C++"
  flags { "StaticRuntime", "FatalWarnings" }

  buildoptions { '/Gm-' }
  buildoptions { '/MP' }
  linkoptions { '/ignore:4006' } -- ignore multiple libraries defining the same symbol

  ignoredefaultlibraries { "msvcrt" }
  
  filter { "configurations:Release" }
    flags { "LinkTimeOptimization" }
  
  filter { }
  
  defines { "_CRT_SECURE_NO_WARNINGS", "SSE2", "GLEW_STATIC" }
  
  objdir "intermediate/obj"

  files { "src/**.c", "src/**.cpp", "src/**.h", "src/**.inl", "src/**rc" }
  files { "include/**.cpp", "include/**.h", "include/**.inl", "src/**rc" }
  files { "project.lua" }
  
  includedirs { "include" }
  includedirs { "include/**" }
  includedirs { "3rdParty" }
  includedirs { "3rdParty/GLFW/include" }
  includedirs { "3rdParty/glew/include" }

  filter { "configurations:Debug", "system:Windows" }
    ignoredefaultlibraries { "libcmt" }
  filter { }
  
  targetname(ProjectName)
  targetdir "lib"
  debugdir "lib"
  
filter {}
configuration {}

linkoptions { "3rdParty/GLFW/lib/glfw3.lib" }
linkoptions { "3rdParty/GLFW/lib/libglew32.lib" }
linkoptions { "opengl32.lib", "glu32.lib" }

warnings "Extra"

filter {"configurations:Release"}
  targetname "%{prj.name}"
filter {"configurations:Debug"}
  targetname "%{prj.name}D"

flags { "NoMinimalRebuild", "NoPCH" }
exceptionhandling "Off"
rtti "Off"
floatingpoint "Fast"

filter { "configurations:Debug*" }
	defines { "_DEBUG" }
	optimize "Off"
	symbols "On"

filter { "configurations:Release" }
	defines { "NDEBUG" }
	optimize "Speed"
	flags { "NoFramePointer", "NoBufferSecurityCheck" }
	symbols "On"

filter { "system:windows", "configurations:Release", "action:vs2012" }
	buildoptions { "/d2Zi+" }

filter { "system:windows", "configurations:Release", "action:vs2013" }
	buildoptions { "/Zo" }

filter { "system:windows", "configurations:Release" }
	flags { "NoIncrementalLink" }

filter {}
  flags { "NoFramePointer", "NoBufferSecurityCheck" }
