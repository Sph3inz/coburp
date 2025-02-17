@echo off
echo Cleaning up...
taskkill /F /IM java.exe /FI "WINDOWTITLE eq Burp Suite*" 2>nul
timeout /t 2 /nobreak >nul

echo Removing target directory...
if exist target (
    rmdir /s /q target
    if errorlevel 1 (
        echo Failed to remove target directory
        echo Please close Burp Suite and any other applications that might be using the files
        pause
        exit /b 1
    )
)

echo Building extension...
call mvn package
if errorlevel 1 (
    echo Build failed
    pause
    exit /b 1
)

echo Build successful!
echo The extension JAR file is located in target/pentest-browser-extension-1.0-SNAPSHOT-jar-with-dependencies.jar
pause 