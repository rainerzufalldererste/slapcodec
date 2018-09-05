@ECHO OFF

ECHO 1. Visual Studio 2017 Solution
ECHO 2. Visual Studio 2015 Solution
ECHO 3. Visual Studio 2013 Solution

CHOICE /N /C:123 /M "[1-3]:"

IF ERRORLEVEL ==3 GOTO THREE
IF ERRORLEVEL ==2 GOTO TWO
IF ERRORLEVEL ==1 GOTO ONE
GOTO END

:THREE
 ECHO Creating VS2013 Project...
 premake\premake5.exe vs2013
 GOTO END

:TWO
 ECHO Creating VS2015 Project...
 premake\premake5.exe vs2015
 GOTO END

:ONE
 ECHO Creating VS2017 Project...
 premake\premake5.exe vs2017
 GOTO END

:END

