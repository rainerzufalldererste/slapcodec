solution "slapcodec"
  
  editorintegration "On"
  configurations { "Debug", "Release" }
  platforms { "x64" }

  dofile "slapcodec/project.lua"
    location("slapcodec")

  dofile "TestApp/project.lua"
    location("TestApp")