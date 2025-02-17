@echo off
echo Setting up Maven environment...

REM Create Maven directory
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

REM Set environment variables
echo Setting environment variables...
setx MAVEN_HOME "C:\Program Files\Apache\maven\apache-maven-3.9.9" /M
setx PATH "%PATH%;%%MAVEN_HOME%%\bin" /M

echo.
echo Maven setup complete! Please restart your terminal/IDE for the changes to take effect.
echo To verify installation, open a new terminal and run: mvn --version 