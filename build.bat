@echo off
echo Building Pentest Browser Extension...

REM Set Java and Maven environment
set "JAVA_HOME=C:\Program Files\Java\jdk-17"
set "PATH=%PATH%;%JAVA_HOME%\bin;C:\Program Files\Apache\maven\apache-maven-3.9.9\bin"

REM Check if Maven is installed
where mvn >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Maven is not installed or not in PATH
    echo Please install Maven and add it to your PATH
    exit /b 1
)

REM Build the project
mvn package -DskipClean

if %ERRORLEVEL% neq 0 (
    echo Build failed
    exit /b 1
)

echo Build successful!
echo The extension JAR file is located in target/pentest-browser-extension-1.0-SNAPSHOT-jar-with-dependencies.jar 