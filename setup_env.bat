@echo off
echo Setting up Java and Maven environment...

REM Find Java installation
for /f "tokens=2*" %%a in ('reg query "HKEY_LOCAL_MACHINE\SOFTWARE\JavaSoft\Java Runtime Environment" /v "CurrentVersion" 2^>nul') do set "JAVA_VERSION=%%b"
for /f "tokens=2*" %%a in ('reg query "HKEY_LOCAL_MACHINE\SOFTWARE\JavaSoft\Java Runtime Environment\%JAVA_VERSION%" /v "JavaHome" 2^>nul') do set "JAVA_PATH=%%b"

if "%JAVA_PATH%"=="" (
    for /f "tokens=2*" %%a in ('reg query "HKEY_LOCAL_MACHINE\SOFTWARE\JavaSoft\JDK" /v "CurrentVersion" 2^>nul') do set "JAVA_VERSION=%%b"
    for /f "tokens=2*" %%a in ('reg query "HKEY_LOCAL_MACHINE\SOFTWARE\JavaSoft\JDK\%JAVA_VERSION%" /v "JavaHome" 2^>nul') do set "JAVA_PATH=%%b"
)

if "%JAVA_PATH%"=="" (
    echo Java installation not found in registry
    echo Please ensure Java is installed correctly
    exit /b 1
)

REM Set JAVA_HOME
echo Setting JAVA_HOME to: %JAVA_PATH%
setx JAVA_HOME "%JAVA_PATH%" /M

REM Create Maven directory if it doesn't exist
mkdir "C:\Program Files\Apache\maven" 2>nul

REM Check if maven zip exists in downloads
if not exist "%USERPROFILE%\Downloads\apache-maven-3.9.9-bin.zip" (
    echo Maven zip file not found in Downloads folder
    echo Please download apache-maven-3.9.9-bin.zip from https://maven.apache.org/download.cgi
    exit /b 1
)

REM Extract Maven
echo Extracting Maven...
powershell -Command "Expand-Archive -Path '%USERPROFILE%\Downloads\apache-maven-3.9.9-bin.zip' -DestinationPath 'C:\Program Files\Apache\maven' -Force"

REM Set MAVEN_HOME
echo Setting MAVEN_HOME...
setx MAVEN_HOME "C:\Program Files\Apache\maven\apache-maven-3.9.9" /M

REM Update PATH
echo Updating PATH...
setx PATH "%PATH%;%JAVA_HOME%\bin;%MAVEN_HOME%\bin" /M

echo.
echo Environment setup complete! Please restart your terminal/IDE for the changes to take effect.
echo.
echo To verify installation, open a new terminal and run:
echo java -version
echo mvn --version 